import torch
import numpy as np
from model import MotionTransformer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

label_map = {
    0: "JumpingJack",
    1: "SoccerJuggling",
    2: "HammerThrow",
    3: "BodyWeightSquats",
    4: "WallPushups",
    5: "TaiChi",
    6: "JumpRope",
    7: "WritingOnBoard",
    8: "HandstandPushups",
    9: "TennisSwing"
}


# Load Data (CORRECT FILES)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


# Normalize (IMPORTANT)
# same stats as training
X_train = np.load("X_train.npy")
mean = X_train.mean()
std = X_train.std()

X_test = (X_test - mean) / std

# Convert to tensor
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long)


model = MotionTransformer(
    input_dim=66,
    d_model=128,
    num_layers=3,
    heads=4,
    num_classes=10,
    seq_len=50
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


# Prediction (BATCH MODE)
with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()


# Accuracy
acc = (preds == y_test.numpy()).mean() * 100
print(f"\n✅ Accuracy: {acc:.2f}%")


# Confusion Matrix
cm = confusion_matrix(y_test.numpy(), preds)

# 🔥 label map ko readable format me convert kiya
labels = [f"{label_map[i]} ({i})" for i in range(len(label_map))]

plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# 🔥 labels rotate (important warna overlap hoga)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

# save 
plt.savefig("confusion_matrix_labeled.png", dpi=300)

plt.show()


# Classification Report
print("\n📊 Classification Report:\n")
print(classification_report(y_test.numpy(), preds))