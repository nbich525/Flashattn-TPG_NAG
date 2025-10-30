import os
import torch
import shutil
from diffusers import StableDiffusionPipeline
from flash_attention_patch import apply_flash_attention_patch  # cần có file này cùng thư mục

# ==== CONFIG ====
BASE_MODEL = "runwayml/stable-diffusion-v1-5"   # model gốc
SAVE_DIR = "/content/drive/MyDrive/Interior-stable-difusion/flash_sd15_full_patched"
# =================

os.makedirs(SAVE_DIR, exist_ok=True)

print("🚀 Đang tải model gốc...")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to("cuda")

# 1️⃣ Patch FlashAttention
print("🔄 Đang patch FlashAttention vào UNet...")
num_replaced = apply_flash_attention_patch(pipe.unet)
print(f"✅ Đã patch FlashAttention cho UNet ({num_replaced} module(s))")

# 2️⃣ Lưu pipeline HuggingFace chuẩn (có thể load bình thường)
print("💾 Đang lưu pipeline chuẩn (diffusers format)...")
pipe.save_pretrained(SAVE_DIR, safe_serialization=False)

# 3️⃣ Lưu riêng state_dict của UNet (để không mất cấu trúc FlashAttention)
unet_state_path = f"{SAVE_DIR}/unet_flashattn_state_dict.pth"
torch.save(pipe.unet.state_dict(), unet_state_path)
print(f"✅ Đã lưu state_dict UNet: {unet_state_path}")

# 4️⃣ Lưu bản sao file patch để auto-load về sau
if os.path.exists("flash_attention_patch.py"):
    shutil.copy("flash_attention_patch.py", f"{SAVE_DIR}/flash_attention_patch.py")
    print("✅ Đã sao chép flash_attention_patch.py vào thư mục model")
else:
    print("⚠️ Không tìm thấy flash_attention_patch.py trong thư mục hiện tại!")

# 5️⃣ Kiểm tra lại số lượng module FlashAttention
flash_count = sum("flash" in m.__class__.__name__.lower() for m in pipe.unet.modules())
print(f"⚡ Tổng số module FlashAttention đang hoạt động: {flash_count}")

print(f"🎉 Hoàn tất! Model đã được lưu đầy đủ tại:\n{SAVE_DIR}")