import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import MotionTransformer
from data_loader import load_array


# Create folder for saved models
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Custom Dataset
class MotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Load Data
X_train = load_array("X_train.npy")
y_train = load_array("y_train.npy")

X_test = load_array("X_test.npy")
y_test = load_array("y_test.npy")

# Normalize
mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# DataLoader
train_loader = DataLoader(MotionDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(MotionDataset(X_test, y_test), batch_size=32)


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MotionTransformer(
    input_dim=66,
    d_model=128,
    num_layers=3,
    heads=4,
    num_classes=10,
    seq_len=50
).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training Loop
epochs = 60
best_acc = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(X)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)

            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Test Acc: {acc:.2f}%")

    # Save model every epoch
    epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), epoch_model_path)

    # Save best model separately
    if acc > best_acc:
        best_acc = acc
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)

print("Best Accuracy:", best_acc)
