#!/usr/bin/env python3
# train_flash_latent_with_embeds_oomsafe_v2.py
# Chunked training — pre-move chunk(latents+embeds) to GPU once per chunk to avoid fragmentation

import os, time, gc, math
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler

# ========== CONFIG ==========
MODEL_DIR = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"
LATENTS_PATH = "/content/drive/MyDrive/Interior-stable-difusion/train_latents_32.pt"
EMB_PATH = "/content/drive/MyDrive/Interior-stable-difusion/train_text_embeds.pt"
OUTPUT_DIR = "/content/drive/MyDrive/Interior-stable-difusion/checkpoints_final_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 1
CHUNK_SIZE = 5         # nhỏ -> an toàn; tăng dần nếu ổn
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8    # accumulate để mô phỏng batch lớn
LR = 1e-5
SAVE_EVERY_CHUNK = True
SAVE_STEP_INTERVAL = 500
PRINT_MEM = True
NUM_WORKERS = 0
# ==========================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Performance knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

def mem(label=""):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()/1e9
    reserved = torch.cuda.memory_reserved()/1e9
    if PRINT_MEM:
        print(f"[MEM] {label:25s} allocated={alloc:.3f}GB reserved={reserved:.3f}GB")

def offload_optimizer_state_to_cpu(optimizer, verbose=False):
    """Move optimizer state tensors to CPU to free GPU memory (best-effort)."""
    try:
        sd = optimizer.state_dict()
        for p_key, p_state in sd.get("state", {}).items():
            for k, v in list(p_state.items()):
                if torch.is_tensor(v) and v.is_cuda:
                    p_state[k] = v.cpu()
        optimizer.load_state_dict(sd)
        if verbose:
            print("[OPT] optimizer state offloaded to CPU")
    except Exception as e:
        if verbose:
            print("[OPT] offload failed:", e)

def dump_diagnostics(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    fn = os.path.join(log_dir, "cuda_memory_summary.txt")
    try:
        with open(fn, "w") as f:
            try:
                f.write(torch.cuda.memory_summary())
            except Exception as e:
                f.write("memory_summary failed: " + repr(e) + "\n")
    except Exception:
        pass

# -------- Load UNet (params float32) ----------
print("Loading UNet (float32 params)...")
unet = UNet2DConditionModel.from_pretrained(MODEL_DIR, subfolder="unet", torch_dtype=torch.float32)
state_path = os.path.join(MODEL_DIR, "unet_flashattn_state_dict.pth")
if os.path.exists(state_path):
    sd = torch.load(state_path, map_location="cpu")
    unet.load_state_dict(sd, strict=False)
    print("Loaded unet_flashattn_state_dict.pth")
unet = unet.to(DEVICE)
if hasattr(unet, "enable_gradient_checkpointing"):
    unet.enable_gradient_checkpointing()
    print("Gradient checkpointing enabled")

scheduler = DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")

# -------- Load latents & embeddings (CPU) ----------
print("Loading latents & embeddings on CPU...")
data = torch.load(LATENTS_PATH, map_location="cpu")
latents_all = data["latents"]   # expected [N,4,32,32]
prompts = data.get("prompts", None)
N = latents_all.shape[0]
print(f"Total latents: {N}")

# Convert latents to float16 on CPU (small memory) to speed transfer
if latents_all.dtype != torch.float16:
    latents_all = latents_all.half()
    print("Converted latents_all -> float16 (CPU)")

emb_data = torch.load(EMB_PATH, map_location="cpu")
embeds_all = emb_data["embeds"]
if embeds_all.dtype != torch.float16:
    embeds_all = embeds_all.half()
print(f"Embeds shape: {embeds_all.shape}")
assert embeds_all.shape[0] == N

# optimizer + scaler
optimizer = optim.AdamW(unet.parameters(), lr=LR)
scaler = GradScaler(enabled=True)
num_timesteps = getattr(scheduler.config, "num_train_timesteps", 1000)
global_step = 0

def get_chunk_indices(total, chunk_size):
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield start, end

print("Begin training (v2) — pre-moving chunk to GPU once per chunk.")
try:
    for epoch in range(EPOCHS):
        for chunk_start, chunk_end in get_chunk_indices(N, CHUNK_SIZE):
            chunk_latents_cpu = latents_all[chunk_start:chunk_end]   # CPU half
            chunk_embeds_cpu  = embeds_all[chunk_start:chunk_end]    # CPU half
            n_samples = chunk_latents_cpu.shape[0]
            if n_samples == 0:
                continue
            print(f"\nEpoch {epoch+1} chunk {chunk_start}:{chunk_end} ({n_samples} samples) — moving chunk to GPU once")

            # MOVE ENTIRE CHUNK TO GPU ONCE (reduces per-batch allocations)
            torch.cuda.empty_cache(); gc.collect()
            mem("before chunk move")
            chunk_latents = chunk_latents_cpu.to(DEVICE, dtype=torch.float16, non_blocking=True)
            chunk_embeds  = chunk_embeds_cpu.to(DEVICE, dtype=torch.float16, non_blocking=True)
            mem("after chunk move")

            unet.train()
            # iterate by manual slicing (no DataLoader) to avoid allocations/pin
            for i in range(0, n_samples, BATCH_SIZE):
                try:
                    optimizer.zero_grad(set_to_none=True)
                    lat_batch = chunk_latents[i:i+BATCH_SIZE]       # already on GPU
                    text_emb  = chunk_embeds[i:i+BATCH_SIZE]       # already on GPU

                    mem("before noise")
                    timesteps = torch.randint(0, num_timesteps, (lat_batch.shape[0],), device=DEVICE, dtype=torch.long)
                    noise = torch.randn_like(lat_batch, device=DEVICE, dtype=torch.float16)

                    with torch.no_grad():
                        # prefer fp16 path; fallback if scheduler needs fp32
                        try:
                            noisy = scheduler.add_noise(lat_batch, noise, timesteps)
                        except Exception:
                            noisy = scheduler.add_noise(lat_batch.to(torch.float32), noise.to(torch.float32), timesteps).to(torch.float16)

                    mem("after noisy")

                    # forward + loss under autocast
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = unet(noisy, timesteps, encoder_hidden_states=text_emb).sample
                        loss = nn.functional.mse_loss(pred, noise)

                    mem("after forward")

                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss")

                    scaler.scale(loss).backward()
                    mem("after backward")

                    # optimizer step when accumulate reached
                    if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
                        grads_exist = any(p.grad is not None for p in unet.parameters() if p.requires_grad)
                        if grads_exist:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        # offload optimizer state to CPU to free GPU memory
                        offload_optimizer_state_to_cpu(optimizer, verbose=False)
                        optimizer.zero_grad(set_to_none=True)
                        mem("after optimizer_step")

                    if global_step % 50 == 0:
                        loss_val = float(loss.detach().cpu())
                        print(f"[LOG] step={global_step} idx={i} loss={loss_val:.6f}")
                        mem("log")

                    global_step += 1

                    if global_step % SAVE_STEP_INTERVAL == 0:
                        ckpt = os.path.join(OUTPUT_DIR, f"unet_step_{global_step}.pth")
                        torch.save(unet.state_dict(), ckpt)
                        print("Saved checkpoint:", ckpt)

                    # free per-iteration temporaries explicitly (pred, loss, noise, noisy)
                    del pred, loss, noise, noisy
                    torch.cuda.empty_cache(); gc.collect()

                except torch.cuda.OutOfMemoryError:
                    ts = int(time.time())
                    dbg_dir = os.path.join(OUTPUT_DIR, f"oom_debug_step_{global_step}_{ts}")
                    print("🔥 CUDA OOM detected — dumping diagnostics to", dbg_dir)
                    dump_diagnostics(dbg_dir)
                    # save CPU copy of model and opt state
                    try:
                        cpu_sd = {k: v.cpu() for k,v in unet.state_dict().items()}
                        torch.save({"step": global_step, "unet_state_cpu": cpu_sd}, os.path.join(dbg_dir, "unet_state_cpu.pth"))
                        opt_sd = optimizer.state_dict()
                        # move optimizer state tensors to CPU
                        for pk,pv in opt_sd.get("state", {}).items():
                            for kk,vv in list(pv.items()):
                                if torch.is_tensor(vv):
                                    pv[kk] = vv.cpu()
                        torch.save(opt_sd, os.path.join(dbg_dir, "optimizer_state_cpu.pth"))
                    except Exception as e:
                        print("Save CPU state failed:", e)
                    torch.cuda.empty_cache(); gc.collect()
                    raise RuntimeError(f"OOM during training at global_step={global_step}. Diagnostics saved to {dbg_dir}")

            # end chunk
            if SAVE_EVERY_CHUNK:
                ckpt = os.path.join(OUTPUT_DIR, f"unet_chunk_{chunk_start}_{chunk_end}.pth")
                torch.save(unet.state_dict(), ckpt)
                print("Saved chunk checkpoint:", ckpt)

            # free chunk tensors on GPU
            del chunk_latents, chunk_embeds, chunk_latents_cpu, chunk_embeds_cpu
            torch.cuda.empty_cache(); gc.collect()
            time.sleep(0.5)

    print("🎉 Training finished successfully")
    torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, "unet_final.pth"))

except KeyboardInterrupt:
    print("Interrupted by user — saving checkpoint...")
    torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"unet_interrupt_step_{global_step}.pth"))
    raise
