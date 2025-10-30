import time

import torch
from diffusers import AutoencoderKL
from pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, device_map = 'auto')
pipe = StableDiffusionXLTPGPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    # custom_pipeline="pipeline.py",
    use_safetensors=True,
    device_map = 'cuda'
)




pipe.load_lora_weights('pytorch_lora_weights.safetensors')

# pipe = apply_enhanced_custom_attention(pipe, qk_norm=True, gate_heads=True, scale=1.0, use_rms_norm=False)

device = "cuda" if torch.cuda.is_available() else "cpu"


# text_encoder
# QWEN_ID = "Qwen/Qwen3-8B"
# tok = AutoTokenizer.from_pretrained(QWEN_ID, use_fast=True)
# qwen = AutoModel.from_pretrained(QWEN_ID, torch_dtype=pipe.unet.dtype).to(device).eval()

# # 3) Build a projection from Qwen3 hidden_size -> SD1.5 text hidden (usually 768)
# target_len = pipe.tokenizer.model_max_length     # typically 77
# target_dim = pipe.text_encoder.config.hidden_size  # typically 768
# proj = torch.nn.Linear(qwen.config.hidden_size, target_dim, bias=False).to(device, dtype=pipe.unet.dtype)

# def qwen_prompt_embeds(prompt: str):
#     t = tok(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=target_len).to(device)
#     with torch.no_grad():
#         h = qwen(**t).last_hidden_state  # [B, L, D_qwen]
#     return proj(h)                       # [B, L, target_dim]

prompt = "a bathroom with a marble bathtub and marble countertop. The bathtub is surrounded by three windows, providing natural light into the room. The bathroom also features a sink situated near the countertop.In addition to the bathroom fixtures, there is a vase placed on the countertop"
# pos = qwen_prompt_embeds(prompt)
# neg = qwen_prompt_embeds("")

# img = pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
#            num_inference_steps=25, guidance_scale=7.0).images[0]
# img

start_time = time.time()

output = pipe(
   prompt,
       width=512,
    height=512,
    num_inference_steps=15,
    guidance_scale=4.0,
    pag_scale=3.0,
    pag_applied_layers=['mid']
).images[0]
# output = pipe(
#     [prompt],
#     width=512,
#     height=512,
#     num_inference_steps=25,
#     guidance_scale=0.0,
#     pag_scale=5.0,
#     pag_applied_layers_index=['m0']
# ).images[0]

# image = pipe(prompt=prompt, num_inference_steps=20).images[0]

# Save the SDXL image output.
output.save('sdxl_output.png')

end_time = time.time()    # ⏱️ kết thúc
print(f"⏳ Thời gian sinh ảnh: {end_time - start_time:.2f} giây")