import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from pathlib import Path


# Uses the current project folder by default, regardless of the Windows user name.
# Set UCF101_DATASET_PATH if the dataset is stored elsewhere.
root_path = Path(os.environ.get("UCF101_DATASET_PATH", Path(__file__).resolve().parent / "UCF-101"))

if not root_path.is_dir():
    raise FileNotFoundError(
        f"UCF-101 dataset not found at {root_path}. Place it there or set UCF101_DATASET_PATH."
    )


# SELECTED FOLDERS ONLY
selected_folders = [
    "JumpingJack",
    "SoccerJuggling",
    "HammerThrow",
    "ThrowDiscus",
    "LongJump",
    "BodyWeightSquats",
    "WritingOnBoard",
    "WallPushups",
    "RopeClimbing",
    "TaiChi",
    "JumpRope",
    "CleanAndJerk",
    "PullUps",
    "Lunges",
    "HandstandPushups",
    "HandstandWalking",
    "GolfSwing",
    "FloorGymnastics",
    "TennisSwing"
]


# MediaPipe setup
mp_pose = mp.solutions.pose

pose_high = mp_pose.Pose(static_image_mode=True, model_complexity=2)
pose_low  = mp_pose.Pose(static_image_mode=True, model_complexity=0)


# FRAME EXTRACTION
def extract_frames(video_path, max_frames=150):
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



# POSE CHECK
def check_video(frames, video_name):
    for idx, frame in enumerate(frames):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = pose_high.process(image)

        if not result.pose_landmarks:
            result = pose_low.process(image)

        if not result.pose_landmarks:
            print(f"❌ {video_name} failed at frame {idx}")
            return False

    return True



# MAIN LOOP (ONLY SELECTED)
print(f"\n🎯 Processing ONLY selected folders\n")

for folder in selected_folders:

    folder_path = os.path.join(root_path, folder)

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder}")
        continue

    good_folder = os.path.join(folder_path, f"good_{folder}")
    bad_folder  = os.path.join(folder_path, f"bekar_{folder}")

    os.makedirs(good_folder, exist_ok=True)
    os.makedirs(bad_folder, exist_ok=True)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

    print(f"\n▶️ {folder} ({len(video_files)} videos)")

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)

        frames = extract_frames(video_path)

        if len(frames) == 0:
            dst = os.path.join(bad_folder, video_file)
        else:
            ok = check_video(frames, video_file)
            dst = os.path.join(good_folder if ok else bad_folder, video_file)

        if os.path.exists(dst):
            os.remove(dst)

        try:
            shutil.move(video_path, dst)
        except Exception as e:
            print(f"Error: {e}")

    print(f"✔️ Done: {folder}")

print("\n🎯 Selected dataset cleaned")
