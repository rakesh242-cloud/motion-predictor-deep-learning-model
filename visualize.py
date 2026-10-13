import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import MotionTransformer
import os


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

# FULL MediaPipe Skeleton (33 points)
connections = [
    # Face
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),

    # Arms
    (11,13),(13,15),
    (12,14),(14,16),

    # Shoulders
    (11,12),

    # Torso
    (11,23),(12,24),
    (23,24),

    # Legs
    (23,25),(25,27),(27,31),
    (24,26),(26,28),(28,32),

    # Feet
    (27,29),(29,31),
    (28,30),(30,32)
]


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load Data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Normalize (same as training)
X_train = np.load("X_train.npy")
mean = X_train.mean()
std = X_train.std()

X_test = (X_test - mean) / std


# Load Model (MATCH TRAINING)
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


# Pick Random Sample
idx = np.random.randint(0, len(X_test))
# idx = 1479
sample = X_test[idx]
true_label = y_test[idx]

print("Sample shape:", sample.shape)
print("🫳 selected index",idx); #1479,2035,3053,2632,2023,2537

# Prediction

with torch.no_grad():
    x = torch.from_numpy(sample).float().unsqueeze(0).to(device)
    out = model(x)
    pred = torch.argmax(out, dim=1).item()

print("\n🎯 RESULT")
print("Actual   :", label_map[true_label])
print("Predicted:", label_map[pred])


# Prepare Animation
fig, ax = plt.subplots()

# Stable axis scaling
x_min = sample[:, 0::2].min()
x_max = sample[:, 0::2].max()
y_min = sample[:, 1::2].min()
y_max = sample[:, 1::2].max()

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_max, y_min)  # invert Y

scat = ax.scatter([], [], s=20)

lines = []
for _ in connections:
    line, = ax.plot([], [], lw=2)
    lines.append(line)

title = ax.set_title("")


# Update function
def update(frame):
    pts = sample[frame]

    x_coords = pts[0::2]
    y_coords = pts[1::2]

    scat.set_offsets(np.c_[x_coords, y_coords])

    for i, (a, b) in enumerate(connections):
        if a < len(x_coords) and b < len(x_coords):
            lines[i].set_data(
                [x_coords[a], x_coords[b]],
                [y_coords[a], y_coords[b]]
            )

    title.set_text(
        f"Frame {frame} | Pred: {label_map[pred]} | True: {label_map[true_label]}"
    )

    return [scat] + lines


# Animate
ani = FuncAnimation(fig, update, frames=len(sample), interval=150)

plt.tight_layout()
# save Gif
save_dir = "some output samples"
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join(save_dir, f"sample_{idx}.gif")

ani.save(file_path, writer='pillow', fps=7)

print(f"✅ GIF saved at: {file_path}")

plt.show()