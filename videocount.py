import os
from pathlib import Path


root_path = Path(os.environ.get("UCF101_DATASET_PATH", Path(__file__).resolve().parent / "UCF-101"))

if not root_path.is_dir():
    raise FileNotFoundError(
        f"UCF-101 dataset not found at {root_path}. Place it there or set UCF101_DATASET_PATH."
    )


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


for folder in selected_folders:

    folder_path = os.path.join(root_path, folder)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder}")
        continue

    good_folder = os.path.join(folder_path, f"good_{folder}")
    bad_folder  = os.path.join(folder_path, f"bekar_{folder}")

    # os.makedirs(good_folder, exist_ok=True)
    # os.makedirs(bad_folder, exist_ok=True)

    video_files1 = [f for f in os.listdir(good_folder) if f.endswith(('.avi', '.mp4'))]
    video_files2 = [f for f in os.listdir(bad_folder) if f.endswith(('.avi', '.mp4'))]

    print(f"\n▶️ {folder} -> ({len(video_files1)} good videos)")
    print(f"▶️ {folder} -> ({len(video_files2)} bekar  videos)")
