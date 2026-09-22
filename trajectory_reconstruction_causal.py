"""
Causal trajectory reconstruction for Task 3.

Unlike the Task 2 reconstruction, this version never uses future
frames to reconstruct the trajectory at the current frame.

Missing detections:
    Forward-fill using the most recent available observation.

Smoothing:
    Trailing rolling average using only the current and previous frames.

Outputs:
    recorded_data/bug_clipX_data_causal.csv
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "recorded_data"

CLIPS = [
    "bug_clip1",
    "bug_clip2",
    "bug_clip3",
    "bug_clip4",
    "bug_clip5",
    "bug_clip6",
]

SMOOTHING_WINDOW = 5


# ============================================================
# CAUSAL ORIENTATION HANDLING
# ============================================================

def causal_orientation(theta_series):
    """
    Reconstruct modulo-180 orientation causally.

    We represent orientation as a doubled angle:
        phi = 2 * theta

    This removes the 180-degree ambiguity.

    Missing values are forward-filled, so no future observation
    is used to estimate an earlier frame.
    """

    theta = theta_series.copy()

    # Forward fill only.
    theta = theta.ffill()

    # If the very beginning of the video has no detection,
    # there is no past information available yet.
    # Leave those entries NaN for now.
    theta_rad = np.deg2rad(
        2.0 * theta
    )

    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    # Trailing smoothing only.
    sin_smooth = (
        pd.Series(sin_theta)
        .rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=False
        )
        .mean()
    )

    cos_smooth = (
        pd.Series(cos_theta)
        .rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=False
        )
        .mean()
    )

    theta_smooth = (
        np.rad2deg(
            np.arctan2(
                sin_smooth,
                cos_smooth
            )
        ) / 2.0
    )

    theta_smooth = (
        theta_smooth % 180.0
    )

    return theta_smooth


# ============================================================
# RECONSTRUCT ONE TRAJECTORY
# ============================================================

def reconstruct_causal(df):

    df = df.copy()

    # --------------------------------------------------------
    # Missing positions
    #
    # Forward fill means that at frame t we can use the most
    # recent known position, but never a future position.
    # --------------------------------------------------------

    df["x_cm"] = (
        df["x_cm"].ffill()
    )

    df["y_cm"] = (
        df["y_cm"].ffill()
    )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    df["theta"] = causal_orientation(
        df["theta"]
    )

    # --------------------------------------------------------
    # Causal position smoothing
    #
    # center=False means the value at frame t uses:
    #
    #     t, t-1, t-2, ...
    #
    # and NEVER t+1, t+2, ...
    # --------------------------------------------------------

    df["x_cm"] = (
        df["x_cm"]
        .rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=False
        )
        .mean()
    )

    df["y_cm"] = (
        df["y_cm"]
        .rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # Drop initial frames for which there was not yet any
    # valid observation to forward-fill.
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "x_cm",
            "y_cm",
            "theta",
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# PROCESS ALL CLIPS
# ============================================================

for clip in CLIPS:

    input_path = os.path.join(
        DATA_DIR,
        f"{clip}_data_raw.csv"
    )

    output_path = os.path.join(
        DATA_DIR,
        f"{clip}_data_causal.csv"
    )

    print(
        f"Processing {clip}..."
    )

    if not os.path.exists(input_path):

        print(
            f"  Missing input: {input_path}"
        )

        continue

    df = pd.read_csv(
        input_path
    )

    reconstructed = reconstruct_causal(
        df
    )

    reconstructed.to_csv(
        output_path,
        index=False
    )

    print(
        f"  Saved {len(reconstructed)} frames "
        f"to {output_path}"
    )


print(
    "\nCausal trajectory reconstruction complete."
)