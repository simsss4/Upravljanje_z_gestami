import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Naloži podatke
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data', 'test')
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

# Normalizacija in dodajanje kanalov
X_test = X_test[..., np.newaxis] / 255.0

# Naloži naučen model
model_path = os.path.join(base_dir, 'model', 'model_drowsy_detector.h5')
model = load_model(model_path)

# Oceni model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n Testna natančnost: {test_acc:.4f}")
print(f" Testna izguba: {test_loss:.4f}")

# Napovej rezultate in prikaži podrobnosti
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Poročilo in zmedna matrika
print("\n Klasifikacijsko poročilo:")
print(classification_report(y_test, y_pred, target_names=["Not Drowsy", "Drowsy"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Drowsy", "Drowsy"], yticklabels=["Not Drowsy", "Drowsy"])
plt.xlabel("Napovedano")
plt.ylabel("Dejansko")
plt.title(" Zmedna matrika")
plt.tight_layout()
plt.show()
