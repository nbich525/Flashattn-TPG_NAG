import os, torch
from diffusers import StableDiffusionPipeline
from torch import nn

# ==== CONFIG ====
MODEL_DIR = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"
device = "cuda"
dtype = torch.float16
# =================

print("🚀 Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype
).to(device)

vae = pipe.vae
text_encoder = pipe.text_encoder
unet = pipe.unet
scheduler = pipe.scheduler

vae.to(dtype)
text_encoder.to(dtype)
unet.to(dtype)

# === TEST 1️⃣: Text Encoder ===
print("\n🔤 Checking Text Encoder...")
tokenizer = pipe.tokenizer
prompt = "A modern interior with natural lighting"
text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
input_ids = text_inputs.input_ids.to(device)

with torch.no_grad():
    text_emb = text_encoder(input_ids)[0]

if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
    raise ValueError("❌ NaN or Inf found in text encoder output!")
print("✅ Text encoder OK:", text_emb.shape)

# === TEST 2️⃣: VAE Encode/Decode ===
print("\n🧩 Checking VAE...")
dummy_image = torch.randn(1, 3, 256, 256, device=device, dtype=dtype).clamp(-1, 1)
with torch.no_grad():
    latent = vae.encode(dummy_image).latent_dist.sample() * 0.18215
    recon = vae.decode(latent / 0.18215).sample

for name, tensor in [("latent", latent), ("recon", recon)]:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"❌ NaN or Inf found in VAE {name}!")
print("✅ VAE OK:", latent.shape, "→", recon.shape)

# === TEST 3️⃣: UNet Forward ===
print("\n🔦 Checking UNet...")
b, c, h, w = latent.shape
t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device, dtype=torch.long)
with torch.no_grad():
    noise_pred = unet(latent, t, encoder_hidden_states=text_emb).sample

if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
    raise ValueError("❌ NaN or Inf found in UNet output!")
print("✅ UNet OK:", noise_pred.shape)

# === TEST 4️⃣: Loss Simulation ===
print("\n📉 Checking Loss computation simulation...")
# Fake ground-truth noise
target = torch.randn_like(noise_pred)
loss_fn = nn.MSELoss()
loss = loss_fn(noise_pred, target)

if not torch.isfinite(loss):
    raise ValueError("❌ Non-finite loss detected!")
print(f"✅ Loss OK: {loss.item():.6f}")

# === SUMMARY ===
print("\n🎯 All checks passed — model ready for fine-tuning!")
