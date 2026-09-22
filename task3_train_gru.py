"""
Task 3 - GRU Motion Prediction Training

Uses reconstructed trajectories from Task 2 to train a GRU that predicts
the robot's displacement one second into the future.

Training:
    bug_clip1
    bug_clip2
    bug_clip3

Validation:
    bug_clip4

Final evaluation clips 5 and 6 are NEVER used here.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "recorded_data"
VIDEO_DIR = "resources"

TRAIN_CLIPS = [
    "bug_clip1",
    "bug_clip2",
    "bug_clip3",
]

VAL_CLIPS = [
    "bug_clip4",
]

HISTORY_SECONDS = 2.0
FUTURE_SECONDS = 1.0

BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_PATH = os.path.join(
    DATA_DIR,
    "task3_gru.pth"
)


# ============================================================
# GET VIDEO FPS
# ============================================================

def get_video_fps(clip):
    video_path = os.path.join(
        VIDEO_DIR,
        f"{clip}.mp4"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return fps


# ============================================================
# ADD MOTION FEATURES
# ============================================================

def add_motion_features(df, fps):
    """
    Add velocity and orientation features.

    Orientation is modulo 180 degrees, so theta is represented
    using sin(2*theta) and cos(2*theta).
    """

    df = df.copy()

    # Velocity in cm / second
    df["vx"] = df["x_cm"].diff() * fps
    df["vy"] = df["y_cm"].diff() * fps

    df["vx"] = df["vx"].fillna(0.0)
    df["vy"] = df["vy"].fillna(0.0)

    # Encode modulo-180 orientation
    theta_rad = np.deg2rad(
        2.0 * df["theta"]
    )

    df["theta_sin"] = np.sin(theta_rad)
    df["theta_cos"] = np.cos(theta_rad)

    return df


# ============================================================
# CREATE SEQUENCES FOR ONE CLIP
# ============================================================

def create_sequences(df, fps):

    history_frames = int(
        round(HISTORY_SECONDS * fps)
    )

    future_frames = int(
        round(FUTURE_SECONDS * fps)
    )

    X = []
    y = []

    # Need enough history before t and enough frames after t.
    for end_idx in range(
        history_frames - 1,
        len(df) - future_frames
    ):

        start_idx = (
            end_idx - history_frames + 1
        )

        future_idx = (
            end_idx + future_frames
        )

        history = df.iloc[
            start_idx:end_idx + 1
        ].copy()

        current = df.iloc[end_idx]
        future = df.iloc[future_idx]

        current_x = current["x_cm"]
        current_y = current["y_cm"]

        future_x = future["x_cm"]
        future_y = future["y_cm"]

        # Position relative to current robot position.
        history["x_rel"] = (
            history["x_cm"] - current_x
        )

        history["y_rel"] = (
            history["y_cm"] - current_y
        )

        features = history[
            [
                "x_rel",
                "y_rel",
                "vx",
                "vy",
                "theta_sin",
                "theta_cos",
            ]
        ].to_numpy(
            dtype=np.float32
        )

        # Target = displacement during next second.
        target = np.array(
            [
                future_x - current_x,
                future_y - current_y,
            ],
            dtype=np.float32
        )

        # Skip any sequence containing invalid values.
        if (
            np.isnan(features).any()
            or
            np.isnan(target).any()
        ):
            continue

        X.append(features)
        y.append(target)

    return X, y


# ============================================================
# LOAD A SET OF CLIPS
# ============================================================

def load_dataset(clips):

    X_all = []
    y_all = []

    for clip in clips:

        print(f"Loading {clip}...")

        path = os.path.join(
            DATA_DIR,
            f"{clip}_data_causal.csv"
        )

        df = pd.read_csv(path)

        fps = get_video_fps(clip)

        print(f"  FPS: {fps:.3f}")

        df = add_motion_features(
            df,
            fps
        )

        X_clip, y_clip = create_sequences(
            df,
            fps
        )

        print(
            f"  Created {len(X_clip)} sequences."
        )

        X_all.extend(X_clip)
        y_all.extend(y_clip)

    X = np.array(
        X_all,
        dtype=np.float32
    )

    y = np.array(
        y_all,
        dtype=np.float32
    )

    return X, y


# ============================================================
# PYTORCH DATASET
# ============================================================

class TrajectoryDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# GRU MODEL
# ============================================================

class TrajectoryGRU(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_size,
            2
        )

    def forward(self, x):

        _, hidden = self.gru(x)

        final_hidden = hidden[-1]

        prediction = self.output(
            final_hidden
        )

        return prediction


# ============================================================
# LOAD DATA
# ============================================================

print("\nCreating training dataset...")

X_train, y_train = load_dataset(
    TRAIN_CLIPS
)

print("\nCreating validation dataset...")

X_val, y_val = load_dataset(
    VAL_CLIPS
)

print("\nDataset shapes:")

print(
    "Training:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)


train_dataset = TrajectoryDataset(
    X_train,
    y_train
)

val_dataset = TrajectoryDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CREATE MODEL
# ============================================================

model = TrajectoryGRU(
    input_size=6,
    hidden_size=HIDDEN_SIZE
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")

print("\nTraining GRU...\n")

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(
        train_loader
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += loss.item()

    val_loss /= len(
        val_loader
    )

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  Saved new best model "
            f"(val loss {best_val_loss:.4f})"
        )


print("\nTraining complete.")

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)