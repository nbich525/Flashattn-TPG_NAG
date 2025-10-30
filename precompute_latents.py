import os, json, torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline

# ========= CONFIG =========
MODEL_DIR = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"
TRAIN_JSON = "/content/drive/MyDrive/Interior-stable-difusion/train_valid.json"
OUTPUT_LATENTS = "/content/drive/MyDrive/Interior-stable-difusion/train_latents_32.pt"
device = "cuda"
BATCH_SIZE = 2  # nhỏ để tránh OOM, có thể tăng nếu GPU >16GB
# ==========================

# 1️⃣ Load pipeline + VAE
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16
).to(device)
pipe.vae.to(torch.float16)
pipe.enable_attention_slicing()

# 2️⃣ Transform: resize ảnh về 256×256 → latent 32×32
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

print(f"📂 Loading dataset: {TRAIN_JSON}")
with open(TRAIN_JSON, "r") as f:
    dataset_json = json.load(f)

latents_list, prompts_list = [], []
batch_images, batch_prompts = [], []

# 3️⃣ Encode theo batch nhỏ để tránh OOM
for item in tqdm(dataset_json, desc="Encoding latents (32×32 from 256×256 images)"):
    image = Image.open(item["image"]).convert("RGB")
    image = transform(image)
    batch_images.append(image)
    batch_prompts.append(item["prompt"])

    # Khi đủ batch, encode
    if len(batch_images) == BATCH_SIZE:
        imgs = torch.stack(batch_images).to(device, dtype=torch.float16)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            latent = pipe.vae.encode(imgs).latent_dist.sample() * 0.18215
        latents_list.append(latent.cpu())
        prompts_list.extend(batch_prompts)
        batch_images, batch_prompts = [], []
        torch.cuda.empty_cache()

# Encode batch cuối cùng (nếu còn)
if batch_images:
    imgs = torch.stack(batch_images).to(device, dtype=torch.float16)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        latent = pipe.vae.encode(imgs).latent_dist.sample() * 0.18215
    latents_list.append(latent.cpu())
    prompts_list.extend(batch_prompts)
    torch.cuda.empty_cache()

# 4️⃣ Gộp tất cả latent lại và lưu
latents = torch.cat(latents_list, dim=0)
torch.save({"latents": latents, "prompts": prompts_list}, OUTPUT_LATENTS)

print(f"✅ Saved latents: {latents.shape} → {OUTPUT_LATENTS}")
