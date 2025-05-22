import os
import random
from PIL import Image, ImageEnhance, ImageOps

# ---- Pravilno določimo koren projekta ----
# __file__ najprej da: 
#   C:/.../Upravljanje_z_gestami/UI/Model 2- Razvoj modela za analizo utrujenosti voznika/augment_images.py
SCRIPT_DIR   = os.path.dirname(__file__)
SRC_ROOT = r"C:\Users\Luka\Documents\GitHub\Upravljanje_z_gestami\Datasets\NTHU_Dataset\train_data"
DST_ROOT = r"C:\Users\Luka\Documents\GitHub\Upravljanje_z_gestami\Datasets\NTHU_Dataset\augmentated_data"

AUG_PER_IMG  = 2  # koliko novih primerkov na original

# Ustvari izhodne mape
for cls in ("drowsy", "notdrowsy"):
    os.makedirs(os.path.join(DST_ROOT, cls), exist_ok=True)

def random_augment(img: Image.Image) -> Image.Image:
    """Naključne, a smiselne transformacije."""
    # rotacija ±15°
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    # premik do ±10% v pixlih
    max_dx = int(0.1 * img.width)
    max_dy = int(0.1 * img.height)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    img = img.transform(
    img.size,
    Image.AFFINE,
    (1, 0, dx, 0, 1, dy),
    resample=Image.BILINEAR
)
    # svetlost 70–130%
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    # zrcaljenje s 50% verjetnostjo
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    return img

# ---- Glavna zanka ----
for cls in ("drowsy", "notdrowsy"):
    src_dir = os.path.join(SRC_ROOT, cls)
    dst_dir = os.path.join(DST_ROOT, cls)

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Mapa ne obstaja: {src_dir!r}")

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        base, ext = os.path.splitext(fname)

        for i in range(AUG_PER_IMG):
            aug = random_augment(img)
            out_name = f"{base}_aug{i+1}{ext}"
            aug.save(os.path.join(dst_dir, out_name))

print("Augmentacija končana. Poglej mapo:", DST_ROOT)
