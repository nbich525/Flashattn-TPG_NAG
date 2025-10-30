# precompute_text_embeddings.py
import os, torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline

# CONFIG
MODEL_DIR = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"
LATENTS_PATH = "/content/drive/MyDrive/Interior-stable-difusion/train_latents_32.pt"
OUT_EMB = "/content/drive/MyDrive/Interior-stable-difusion/train_text_embeds.pt"
DEVICE = "cuda"
BATCH = 64  # adjust if GPU mem limited when computing embeddings

# load prompts from latents file
data = torch.load(LATENTS_PATH, map_location="cpu")
prompts = data["prompts"]
N = len(prompts)
print(f"Found {N} prompts")

# load tokenizer + text encoder (on GPU for speed)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR, subfolder="text_encoder").to(DEVICE).eval()

all_embeds = []
with torch.no_grad():
    for i in tqdm(range(0, N, BATCH), desc="Embedding prompts"):
        batch_prompts = prompts[i:i+BATCH]
        tokenized = tokenizer(batch_prompts, padding="max_length", truncation=True,
                              max_length=tokenizer.model_max_length, return_tensors="pt")
        input_ids = tokenized.input_ids.to(DEVICE)
        emb = text_encoder(input_ids)[0]            # shape [B, 77, 768]
        emb = emb.to(torch.float16).cpu()          # store in float16 on CPU
        all_embeds.append(emb)

# concat and save as single tensor (N,77,768)
embeds = torch.cat(all_embeds, dim=0)
torch.save({"embeds": embeds, "prompts": prompts}, OUT_EMB)
print(f"Saved embeddings to {OUT_EMB}, shape {embeds.shape}")

