# flash_attention_patch.py
# Robust, safer FlashAttention wrapper for diffusers UNet attention modules.
# - Uses torch.scaled_dot_product_attention fallback when inputs are not fp16/bf16
# - Validates flash-attn outputs (finite) and falls back to original forward on failure
# - Tries multiple flash-attn call signatures, but never calls them with unsafe dtypes
# - More defensive handling of head_dim / heads and to_out projection

import torch
import torch.nn as nn
import inspect
import warnings

# Try to import flash-attn C entrypoints (may vary by installation)
_flash_funcs = {}
_ffi = None
try:
    import flash_attn.flash_attn_interface as _ffi
    for name in dir(_ffi):
        obj = getattr(_ffi, name)
        if inspect.isfunction(obj):
            _flash_funcs[name] = obj
except Exception:
    _ffi = None
    # silent: flash-attn not available

def _torch_scaled_dot_attn_from_qkv(q, k, v, causal=False):
    """
    Fallback implementation using torch.scaled_dot_product_attention.
    q,k,v shapes: (B, S, H, D)
    Returns out shape: (B, S, H, D)
    """
    B, S, H, D = q.shape
    # reshape to (B*H, S, D)
    q2 = q.permute(0,2,1,3).reshape(B*H, S, D)
    k2 = k.permute(0,2,1,3).reshape(B*H, k.shape[1], D)
    v2 = v.permute(0,2,1,3).reshape(B*H, v.shape[1], D)
    out2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=causal)
    out = out2.view(B, H, S, D).permute(0,2,1,3)
    return out

def _call_flash_attn_safe(q, k, v):
    """
    Try known flash-attn functions in safe manner.
    Only attempts flash-attn calls if dtype is fp16 or bfloat16.
    If no suitable flash-attn function works, return None to indicate fallback.
    """
    # require flash-attn lib present
    if _ffi is None or not _flash_funcs:
        return None

    # only call flash-attn when inputs are low-precision types
    if q.dtype not in (torch.float16, torch.bfloat16):
        return None

    # ensure contiguous to satisfy some implementations
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    # candidate function names seen in the wild
    candidates = [
        "flash_attn_func",
        "flash_attn_unpadded_qkvpacked_func",
        "flash_attn_unpadded_kvpacked_func",
        "flash_attn_unpadded_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_func",
        "flash_attn",
    ]

    # Try each candidate with a few call signatures (best-effort)
    for name in candidates:
        fn = _flash_funcs.get(name)
        if fn is None:
            continue
        try:
            # Pattern A: fn(q, k, v, dropout_p, causal)
            try:
                out = fn(q_c, k_c, v_c, 0.0, False)
                return out
            except TypeError:
                pass
            # Pattern B: fn(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal)
            try:
                qkv = torch.stack([q_c, k_c, v_c], dim=2)  # B, S, 3, H, D
                out = fn(qkv, None, q.shape[1], 0.0, None, False)
                return out
            except TypeError:
                pass
            # Pattern C: fn(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale)
            try:
                qkv = torch.stack([q_c, k_c, v_c], dim=2)
                out = fn(qkv, None, q.shape[1], 0.0, None)
                return out
            except TypeError:
                pass
            # Pattern D: named args
            try:
                out = fn(q=q_c, k=k_c, v=v_c, dropout_p=0.0, causal=False)
                return out
            except TypeError:
                pass
            # Pattern E: simple positional (q,k,v)
            try:
                out = fn(q_c, k_c, v_c)
                return out
            except Exception:
                pass
        except Exception:
            # this flash-attn function failed — try next
            continue

    # no flash-attn function succeeded
    return None

class FlashCrossAttention(nn.Module):
    """
    Wrapper that attempts to use flash-attn where possible, else falls back to original forward.
    Designed to replace attention modules in diffusers' UNet.
    """
    def __init__(self, original_module):
        super().__init__()
        self.orig = original_module
        # try to reuse common linear layers
        self.to_q = getattr(original_module, "to_q", None)
        self.to_k = getattr(original_module, "to_k", None)
        self.to_v = getattr(original_module, "to_v", None)
        self.to_out = getattr(original_module, "to_out", None)
        self._fallback_forward = getattr(original_module, "forward", None)

        # try to infer heads and head_dim safely
        self.heads = getattr(original_module, "num_heads", None)
        if self.heads is None:
            self.heads = getattr(original_module, "heads", None)
        # attempt to read head_dim if present
        self.head_dim = getattr(original_module, "head_dim", None)
        # infer dim from to_q if possible
        self.dim = None
        if self.to_q is not None and hasattr(self.to_q, "in_features"):
            self.dim = self.to_q.in_features
        elif self.to_q is not None and hasattr(self.to_q, "weight"):
            # fallback: use weight shape
            w = getattr(self.to_q, "weight", None)
            if isinstance(w, torch.Tensor):
                self.dim = w.shape[1]
        # final fallback defaults
        if self.heads is None:
            # common SD heads choices: 8, 16
            self.heads = 8
        if self.head_dim is None and self.dim is not None:
            # ensure integer division
            if self.dim % self.heads == 0:
                self.head_dim = self.dim // self.heads
            else:
                # fallback conservative guess
                self.head_dim = max(1, self.dim // self.heads)
        # nothing fatal here; we'll try to be permissive in forward

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Prefer original forward if we cannot build q/k/v
        try:
            # compute q,k,v using stored linear layers
            q = self.to_q(hidden_states) if self.to_q is not None else None
            context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            k = self.to_k(context) if self.to_k is not None else None
            v = self.to_v(context) if self.to_v is not None else None

            # If any of q/k/v is None, fall back to original forward
            if q is None or k is None or v is None:
                if self._fallback_forward is not None:
                    return self._fallback_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
                raise RuntimeError("Cannot obtain q/k/v tensors and no fallback forward available.")

            B, S, total_dim = q.shape
            # determine heads and head_dim
            heads = self.heads
            head_dim = self.head_dim
            if head_dim is None:
                # try compute
                if total_dim % heads == 0:
                    head_dim = total_dim // heads
                else:
                    # last resort: set head_dim = total_dim // heads (floor)
                    head_dim = max(1, total_dim // heads)

            # reshape q,k,v to (B, S, H, D)
            try:
                q = q.view(B, S, heads, head_dim)
                k = k.view(B, k.shape[1], heads, head_dim)
                v = v.view(B, v.shape[1], heads, head_dim)
            except Exception:
                # reshape failed — use fallback forward
                if self._fallback_forward is not None:
                    return self._fallback_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
                raise

            # If inputs are not half or bfloat16, prefer torch fallback for safety
            if q.dtype not in (torch.float16, torch.bfloat16):
                out = _torch_scaled_dot_attn_from_qkv(q, k, v, causal=False)
            else:
                # try flash-attn functions safely
                out = _call_flash_attn_safe(q, k, v)
                if out is None:
                    # flash-attn not available or not used for this dtype -> torch fallback
                    out = _torch_scaled_dot_attn_from_qkv(q, k, v, causal=False)
                else:
                    # ensure output finite — if not, fallback
                    if not torch.isfinite(out).all():
                        # Don't spam logs in normal runs; use warnings
                        warnings.warn("[FlashCrossAttention] flash-attn returned non-finite values — falling back to torch scaled_dot_product_attention.")
                        out = _torch_scaled_dot_attn_from_qkv(q, k, v, causal=False)

            # out shape: (B, S, H, D) or maybe (B, S, H*D) depending on implementation
            if out.ndim == 4:
                B2, S2, H2, D2 = out.shape
                # reshape to (B, S, H*D)
                out = out.reshape(B2, S2, H2 * D2)
            elif out.ndim == 3:
                # assume (B, S, H*D) already
                pass
            else:
                # unexpected shape -> fallback
                if self._fallback_forward is not None:
                    return self._fallback_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
                raise RuntimeError("flash-attn produced output with unexpected ndim")

            # final projection out
            if self.to_out is not None:
                # some modules have to_out as Sequential([Linear, Dropout]) or Linear
                try:
                    if isinstance(self.to_out, nn.Sequential):
                        return self.to_out(out)
                    else:
                        return self.to_out(out)
                except Exception:
                    # if projection fails, try fallback forward
                    if self._fallback_forward is not None:
                        return self._fallback_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
                    raise
            else:
                return out
        except Exception as e:
            # Last resort: try original forward if present
            if self._fallback_forward is not None:
                try:
                    return self._fallback_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
                except Exception:
                    # raise the original error if fallback also fails
                    raise e
            raise

def apply_flash_attention_patch(unet):
    """
    Replace attention-like modules in diffusers UNet with FlashCrossAttention wrappers.
    Returns number of modules replaced.
    """
    replaced = 0
    for name, module in list(unet.named_modules()):
        # Heuristic: modules that have to_q,to_k,to_v attributes are attention modules in diffusers
        if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"):
            # find parent and attribute name to set
            path = name.split(".")
            parent = unet
            for p in path[:-1]:
                parent = getattr(parent, p)
            attr = path[-1]
            try:
                wrapped = FlashCrossAttention(module)
                setattr(parent, attr, wrapped)
                replaced += 1
            except Exception:
                # skip replacement if any unexpected error
                continue
    print(f"✅ Đã kích hoạt FlashAttention cho UNet (thay {replaced} module(s))")
    return replaced
