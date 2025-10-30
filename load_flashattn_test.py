from diffusers import StableDiffusionPipeline
import torch

# ✅ Đường dẫn model đã patch FlashAttention v1
model_dir = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"

# ✅ Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16
).to("cuda")

# ✅ Prompt để test
prompt = "a spacious bathroom with a white tub, a sink, and a mirror. The bathroom is decorated in neutral tones, giving it a clean and elegant appearance. The bathroom features a large window, allowing natural light to fill the space.There are several decorative elements in the bathroom"

# ✅ Sinh ảnh và lưu ra file
image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
save_path = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched/flashattn_infer_test.png"
image.save(save_path)

print(f"✅ Ảnh đã được lưu tại: {save_path}")
