
import os
import torch
from diffusers import StableDiffusionPipeline
from flash_attention_patch import apply_flash_attention_patch  # bản copy có sẵn trong folder model

# ==== CONFIG ====
MODEL_DIR = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"
UNET_STATE_PATH = f"{MODEL_DIR}/unet_flashattn_state_dict.pth"
# =================

# Kiểm tra file cần thiết
assert os.path.exists(MODEL_DIR), f"❌ Không tìm thấy model tại {MODEL_DIR}"
assert os.path.exists(UNET_STATE_PATH), f"❌ Không tìm thấy file UNet state_dict: {UNET_STATE_PATH}"

print("🚀 Đang tải pipeline từ Hugging Face format...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16
).to("cuda")

# 1️⃣ Patch lại FlashAttention trước khi load state_dict
print("🔄 Đang patch lại FlashAttention vào UNet...")
num_replaced = apply_flash_attention_patch(pipe.unet)
print(f"✅ Đã kích hoạt FlashAttention cho UNet ({num_replaced} module(s))")

# 2️⃣ Load lại trọng số (đảm bảo đúng cấu trúc FlashAttention)
print("💾 Đang load state_dict vào UNet...")
state_dict = torch.load(UNET_STATE_PATH, map_location="cuda")
missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)

if missing:
    print(f"⚠️ Thiếu {len(missing)} keys (không ảnh hưởng): {missing[:3]}...")
if unexpected:
    print(f"⚠️ Có {len(unexpected)} keys lạ: {unexpected[:3]}...")

# 3️⃣ Kiểm tra lại các module FlashAttention
flash_modules = [
    name for name, module in pipe.unet.named_modules()
    if "flash" in module.__class__.__name__.lower()
]
print(f"⚡ Số module FlashAttention đang hoạt động: {len(flash_modules)}")

if flash_modules:
    print("✅ FlashAttention đã được load thành công trong UNet!")
else:
    print("❌ Không phát hiện FlashAttention!")

# 4️⃣ Kiểm tra thử pipeline hoạt động
print("🧪 Kiểm tra thử inference 1 ảnh nhỏ (tùy chọn)...")
prompt = "a modern interior design, natural light, cozy atmosphere"
image = pipe(prompt, num_inference_steps=5, height=256, width=256).images[0]
image.save(f"{MODEL_DIR}/test_flashattention_output.png")
print("✅ Đã tạo ảnh test: test_flashattention_output.png")

print("🎉 Model đã sẵn sàng với FlashAttention!")

