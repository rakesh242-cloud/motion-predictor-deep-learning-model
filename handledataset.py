import os
import cv2
import numpy as np
import random
import mediapipe as mp
from sklearn.model_selection import train_test_split


DATASET_PATH = r"C:\ml project course\motion-pattern prediction model\UCF-101"

ACTIONS = {
    "JumpingJack": 0,
    "SoccerJuggling": 1,
    "HammerThrow": 2,
    "BodyWeightSquats": 3,
    "WallPushups": 4,
    "TaiChi": 5,
    "JumpRope": 6,
    "WritingOnBoard": 7,
    "HandstandPushups": 8,
    "TennisSwing": 9
}

SEQ_LEN = 50
MAX_VIDEOS = 40


# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Frame extraction
def extract_frames(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)

    cap.release()
    return frames


# Pose extraction
def extract_keypoints(frames):
    seq = []

    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img)

        if result.pose_landmarks:
            pts = []
            for lm in result.pose_landmarks.landmark:
                pts.extend([lm.x, lm.y])
            seq.append(pts)

    return np.array(seq)


# Create sequences
def create_sequences(data, seq_len=50):
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i:i+seq_len])
    return seqs

# MAIN PIPELINE
X_train, y_train = [], []
X_test, y_test = [], []

for action in ACTIONS:
    print(f"\nProcessing {action}...")

    good_folder = os.path.join(DATASET_PATH, action, f"good_{action}")
    videos = os.listdir(good_folder)

    # 🔥 random select max 40
    if len(videos) > MAX_VIDEOS:
        videos = random.sample(videos, MAX_VIDEOS)

    # 🔥 VIDEO LEVEL SPLIT
    train_videos, test_videos = train_test_split(
        videos, test_size=0.2, random_state=42
    )

    # ---------------- TRAIN ----------------
    for video in train_videos:
        path = os.path.join(good_folder, video)

        frames = extract_frames(path)
        keypoints = extract_keypoints(frames)

        if len(keypoints) < SEQ_LEN:
            continue

        seqs = create_sequences(keypoints, SEQ_LEN)

        for s in seqs:
            X_train.append(s)
            y_train.append(ACTIONS[action])

    # ---------------- TEST ----------------
    for video in test_videos:
        path = os.path.join(good_folder, video)

        frames = extract_frames(path)
        keypoints = extract_keypoints(frames)

        if len(keypoints) < SEQ_LEN:
            continue

        seqs = create_sequences(keypoints, SEQ_LEN)

        for s in seqs:
            X_test.append(s)
            y_test.append(ACTIONS[action])


# Convert to numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)


# Save
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\n✅ Dataset saved (NO LEAKAGE)")