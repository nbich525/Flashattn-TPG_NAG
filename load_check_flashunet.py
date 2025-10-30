python3 - <<'PYCODE'
from diffusers import StableDiffusionPipeline
import torch

print("🐍 Đang dùng môi trường Python:", torch.__version__)
print("🔥 CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))

# ✅ Load pipeline đã patch FlashAttention
model_dir = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full"
print(f"🔄 Đang load pipeline từ: {model_dir}")

pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16
).to("cuda")

# ✅ Test generate ảnh
prompt = "modern minimalist living room, warm sunlight, wooden floor, soft lighting, high detail"
print("🎨 Đang sinh ảnh...")
image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

output_path = "/content/drive/MyDrive/Interior-stable-difusion/test_flashattn.png"
image.save(output_path)
print(f"✅ Ảnh đã sinh và lưu thành công: {output_path}")
PYCODE
