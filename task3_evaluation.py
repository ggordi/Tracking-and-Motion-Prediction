"""
Task 3 Evaluation

Compares:
    1. GRU motion prediction
    2. Stationary baseline
    3. Constant-velocity baseline

All methods are evaluated against the same manually annotated
t + 1 second endpoints.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "recorded_data"
VIDEO_DIR = "resources"

MODEL_PATH = os.path.join(
    DATA_DIR,
    "task3_gru.pth"
)

ANNOTATION_PATH = os.path.join(
    DATA_DIR,
    "manual_task3_annotations.csv"
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "task3_evaluation_results.csv"
)

PLOT_PATH = os.path.join(
    DATA_DIR,
    "task3_prediction_errors.png"
)

HISTORY_SECONDS = 2.0

HIDDEN_SIZE = 64

VELOCITY_WINDOW_SECONDS = 0.33


# ============================================================
# MODEL
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

        return self.output(
            hidden[-1]
        )


model = TrajectoryGRU(
    input_size=6,
    hidden_size=HIDDEN_SIZE
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# HELPERS
# ============================================================

def get_video_fps(clip):

    path = os.path.join(
        VIDEO_DIR,
        f"{clip}.mp4"
    )

    cap = cv2.VideoCapture(path)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    return fps


def add_motion_features(
    df,
    fps
):

    df = df.copy()

    df["vx"] = (
        df["x_cm"].diff() * fps
    )

    df["vy"] = (
        df["y_cm"].diff() * fps
    )

    df["vx"] = (
        df["vx"].fillna(0.0)
    )

    df["vy"] = (
        df["vy"].fillna(0.0)
    )

    theta_rad = np.deg2rad(
        2.0 * df["theta"]
    )

    df["theta_sin"] = np.sin(
        theta_rad
    )

    df["theta_cos"] = np.cos(
        theta_rad
    )

    return df


def euclidean_error(
    pred_x,
    pred_y,
    gt_x,
    gt_y
):

    return np.sqrt(
        (pred_x - gt_x) ** 2
        +
        (pred_y - gt_y) ** 2
    )


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

annotations = pd.read_csv(
    ANNOTATION_PATH
)

results = []


# ============================================================
# EVALUATE
# ============================================================

for video_name in annotations[
    "video"
].unique():

    clip = os.path.splitext(
        video_name
    )[0]

    print(
        f"Evaluating {clip}..."
    )

    fps = get_video_fps(
        clip
    )

    history_frames = int(
        round(
            HISTORY_SECONDS * fps
        )
    )

    velocity_window = int(
        round(
            VELOCITY_WINDOW_SECONDS
            * fps
        )
    )

    path = os.path.join(
        DATA_DIR,
        f"{clip}_data_interpolated.csv"
    )

    df = pd.read_csv(path)

    df = add_motion_features(
        df,
        fps
    )

    clip_annotations = annotations[
        annotations["video"]
        == video_name
    ]

    for _, annotation in (
        clip_annotations.iterrows()
    ):

        start_frame = int(
            annotation["start_frame"]
        )

        gt_x = annotation["x_GT"]
        gt_y = annotation["y_GT"]

        # Locate start frame in trajectory.
        matches = df.index[
            df["frame"]
            == start_frame
        ]

        if len(matches) == 0:

            print(
                f"Could not find frame "
                f"{start_frame}"
            )

            continue

        end_idx = matches[0]

        if (
            end_idx
            < history_frames - 1
        ):
            continue

        current = df.iloc[
            end_idx
        ]

        current_x = current[
            "x_cm"
        ]

        current_y = current[
            "y_cm"
        ]

        # ----------------------------------------------------
        # 1. Stationary baseline
        # ----------------------------------------------------

        stationary_x = current_x
        stationary_y = current_y

        # ----------------------------------------------------
        # 2. Constant velocity baseline
        # ----------------------------------------------------

        velocity_start_idx = max(
            0,
            end_idx - velocity_window
        )

        old = df.iloc[
            velocity_start_idx
        ]

        elapsed_frames = (
            end_idx
            - velocity_start_idx
        )

        elapsed_time = (
            elapsed_frames / fps
        )

        if elapsed_time > 0:

            vx = (
                current_x
                - old["x_cm"]
            ) / elapsed_time

            vy = (
                current_y
                - old["y_cm"]
            ) / elapsed_time

        else:

            vx = 0.0
            vy = 0.0

        constant_velocity_x = (
            current_x + vx
        )

        constant_velocity_y = (
            current_y + vy
        )

        # ----------------------------------------------------
        # 3. GRU
        # ----------------------------------------------------

        history = df.iloc[
            end_idx
            - history_frames
            + 1
            :
            end_idx + 1
        ].copy()

        history["x_rel"] = (
            history["x_cm"]
            - current_x
        )

        history["y_rel"] = (
            history["y_cm"]
            - current_y
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

        X = torch.tensor(
            features,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            displacement = (
                model(X)
                .squeeze(0)
                .numpy()
            )

        gru_x = (
            current_x
            + displacement[0]
        )

        gru_y = (
            current_y
            + displacement[1]
        )

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        stationary_error = (
            euclidean_error(
                stationary_x,
                stationary_y,
                gt_x,
                gt_y
            )
        )

        velocity_error = (
            euclidean_error(
                constant_velocity_x,
                constant_velocity_y,
                gt_x,
                gt_y
            )
        )

        gru_error = (
            euclidean_error(
                gru_x,
                gru_y,
                gt_x,
                gt_y
            )
        )

        results.append({
            "video": video_name,
            "start_frame": start_frame,

            "x_GT": gt_x,
            "y_GT": gt_y,

            "stationary_x": stationary_x,
            "stationary_y": stationary_y,

            "velocity_x": constant_velocity_x,
            "velocity_y": constant_velocity_y,

            "gru_x": gru_x,
            "gru_y": gru_y,

            "stationary_error_cm":
                stationary_error,

            "velocity_error_cm":
                velocity_error,

            "gru_error_cm":
                gru_error,
        })


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    results
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


def print_stats(
    name,
    column
):

    print(
        f"\n{name}"
    )

    print("-" * 50)

    print(
        f"Mean:   "
        f"{results[column].mean():.3f} cm"
    )

    print(
        f"Median: "
        f"{results[column].median():.3f} cm"
    )

    print(
        f"90th:   "
        f"{results[column].quantile(0.90):.3f} cm"
    )


print("\n")
print("=" * 60)
print("TASK 3 EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nEvaluation points: "
    f"{len(results)}"
)

print_stats(
    "STATIONARY",
    "stationary_error_cm"
)

print_stats(
    "CONSTANT VELOCITY",
    "velocity_error_cm"
)

print_stats(
    "GRU",
    "gru_error_cm"
)


# ============================================================
# ERROR PLOT
# ============================================================

plot_data = [
    results["stationary_error_cm"],
    results["velocity_error_cm"],
    results["gru_error_cm"],
]

plt.figure(
    figsize=(8, 5)
)

plt.boxplot(
    plot_data,
    tick_labels=[
        "Stationary",
        "Constant Velocity",
        "GRU",
    ]
)

plt.ylabel(
    "Prediction Error (cm)"
)

plt.title(
    "Task 3: One-Second Prediction Error"
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=200
)

plt.show()


print(
    f"\nSaved results to: "
    f"{OUTPUT_PATH}"
)

print(
    f"Saved plot to: "
    f"{PLOT_PATH}"
)