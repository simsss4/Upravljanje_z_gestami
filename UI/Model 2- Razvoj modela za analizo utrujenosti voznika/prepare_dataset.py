import os
import numpy as np
from PIL import Image

# Absolutna pot
input_dir = "Datasets/NTHU_Dataset/train_data"
output_dir = "UI/Model 2- Razvoj modela za analizo utrujenosti voznika/data"
classes = ["drowsy", "notdrowsy"]
image_size = (128, 128)

def prepare_dataset():
    os.makedirs(output_dir, exist_ok=True)
    X, y = [], []

    for label, cls in enumerate(classes):
        cls_dir = os.path.join(input_dir, cls)
        print(f"➡️ Obdelujem razred: {cls} (iz {cls_dir})")
        if not os.path.exists(cls_dir):
            print(f"⚠️ Napaka: Pot {cls_dir} ne obstaja.")
            continue

        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath).convert("L")
                    img = img.resize(image_size, Image.Resampling.LANCZOS)
                    X.append(np.array(img))
                    y.append(label)
                    if len(X) % 500 == 0:
                        print(f"  ✅ Obdelanih {len(X)} slik...")
                except Exception as e:
                    print(f"❌ Napaka pri sliki {fpath}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"📦 Shrani {X.shape[0]} vzorcev ({X.shape[1]}x{X.shape[2]})")

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print("✅ Dataset uspešno pripravljen in shranjen.")

if __name__ == "__main__":
    prepare_dataset()
