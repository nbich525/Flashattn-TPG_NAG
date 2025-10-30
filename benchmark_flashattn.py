import torch, time
from diffusers import StableDiffusionPipeline
from torch import autocast

device = "cuda"

# === Cấu hình model paths ===
MODELS = {
    "CompVis_fp32": "runwayml/stable-diffusion-v1-5",
    "CompVis_autocast": "runwayml/stable-diffusion-v1-5",
    "Diffusers_unoptimized": "runwayml/stable-diffusion-v1-5",
    "Diffusers_fp16": "runwayml/stable-diffusion-v1-5",
    "Diffusers_traced": "runwayml/stable-diffusion-v1-5",
    "Diffusers_FlashAttn_v1": "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"
}

prompt = "A cozy living room with natural light, rendered in realistic style"

results = {}

def benchmark(pipe, name, n=10):
    torch.cuda.empty_cache()
    _ = pipe(prompt).images[0]  # warmup
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(n):
        _ = pipe(prompt).images[0]

    torch.cuda.synchronize()
    end = time.time()
    throughput = n / (end - start)
    results[name] = throughput
    print(f"{name:25s}: {throughput:.3f} img/s")

# === CompVis fp32 ===
pipe = StableDiffusionPipeline.from_pretrained(MODELS["CompVis_fp32"], torch_dtype=torch.float32).to(device)
benchmark(pipe, "CompVis_fp32")
del pipe

# === CompVis autocast (fp16 runtime) ===
pipe = StableDiffusionPipeline.from_pretrained(MODELS["CompVis_autocast"], torch_dtype=torch.float32).to(device)
with autocast("cuda"):
    benchmark(pipe, "CompVis_autocast")
del pipe

# === Diffusers unoptimized (fp32) ===
pipe = StableDiffusionPipeline.from_pretrained(MODELS["Diffusers_unoptimized"], torch_dtype=torch.float32).to(device)
benchmark(pipe, "Diffusers_unoptimized")
del pipe

# === Diffusers fp16 ===
pipe = StableDiffusionPipeline.from_pretrained(MODELS["Diffusers_fp16"], torch_dtype=torch.float16).to(device)
benchmark(pipe, "Diffusers_fp16")
del pipe


# === Diffusers + FlashAttention v1 ===
pipe = StableDiffusionPipeline.from_pretrained(MODELS["Diffusers_FlashAttn_v1"], torch_dtype=torch.float16).to(device)
benchmark(pipe, "Diffusers_FlashAttn_v1")
del pipe

# === Tổng kết ===
print("\n=== Benchmark Results ===")
for k, v in results.items():
    print(f"{k:25s}: {v:.3f} img/s")
