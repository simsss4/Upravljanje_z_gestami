import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset():
    """
    Razdeli že pripravljene .npy datoteke na učne, validacijske in testne množice
    in jih shrani v podmape 'train', 'val', 'test' znotraj 'data' mape. 
    Prav tako ustvari podmapo 'raw' in vanjo premakne originalne X.npy in y.npy.
    """
    # Poenostavi poti glede na lokacijo tega skripta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, "data")
    raw_dir    = os.path.join(data_dir, "raw")
    train_dir  = os.path.join(data_dir, "train")
    val_dir    = os.path.join(data_dir, "val")
    test_dir   = os.path.join(data_dir, "test")

    # 1) Ustvari vse potrebne mape
    for d in (raw_dir, train_dir, val_dir, test_dir):
        os.makedirs(d, exist_ok=True)

    # 2) Premakni surove .npy datoteke v 'raw'
    for fname in ("X.npy", "y.npy"):
        src = os.path.join(data_dir, fname)
        dst = os.path.join(raw_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)

    # 3) Naloži podatke iz 'raw'
    X = np.load(os.path.join(raw_dir, "X.npy"))
    y = np.load(os.path.join(raw_dir, "y.npy"))

    # 4) Razdeli: 70% train, 30% temp (za val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)

    # 5) Razdeli temp na 50% val, 50% test (kar pomeni po 15% iz originala)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # 6) Shrani dobljene množice
    np.save(os.path.join(train_dir, "X_train.npy"), X_train)
    np.save(os.path.join(train_dir, "y_train.npy"), y_train)
    np.save(os.path.join(val_dir, "X_val.npy"), X_val)
    np.save(os.path.join(val_dir, "y_val.npy"), y_val)
    np.save(os.path.join(test_dir, "X_test.npy"), X_test)
    np.save(os.path.join(test_dir, "y_test.npy"), y_test)

    # 7) Izpis obsega posamezne množice
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")

if __name__ == "__main__":
    split_dataset()
