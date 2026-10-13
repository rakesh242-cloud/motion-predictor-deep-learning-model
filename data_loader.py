"""Load the processed motion arrays directly from their shared Drive files."""

from io import BytesIO
import os
from pathlib import Path

import gdown
import numpy as np


DRIVE_FILE_IDS = {
    "X_train.npy": "1aJ-cKth-16HsUhO7WBGVhBQc_FpsJN67",
    "y_train.npy": "1BYGsCQUxtvb5dVfLh2lCEdLQGhfvHtph",
    "X_test.npy": "1-APeCVaV5Wx3r4KMmaEkKbPGWmjNIivm",
    "y_test.npy": "19Dk40hlC4PFvT-MGl8gVu0Gpsl_E_KD6",
}


def load_array(filename):
    """Use Drive by default; set MOTION_DATA_SOURCE=local for local .npy files."""
    if filename not in DRIVE_FILE_IDS:
        raise ValueError(f"Unknown dataset file: {filename}")

    source = os.environ.get("MOTION_DATA_SOURCE", "drive").lower()
    if source == "local":
        return np.load(Path(__file__).resolve().parent / filename, allow_pickle=False)
    if source != "drive":
        raise ValueError("MOTION_DATA_SOURCE must be 'drive' or 'local'")

    print(f"Loading {filename} from Google Drive into memory...")
    buffer = BytesIO()
    result = gdown.download(
        id=DRIVE_FILE_IDS[filename], output=buffer, quiet=True, use_cookies=False
    )
    if result is None:
        raise RuntimeError(f"Could not load {filename} from Google Drive")
    buffer.seek(0)
    return np.load(buffer, allow_pickle=False)
