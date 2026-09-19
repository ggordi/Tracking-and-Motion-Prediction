# code for task 2, interpolation and smoothing for trajectory reconstruction

import os
import numpy as np
import pandas as pd


# --------------------------------------------------
# Setup
# --------------------------------------------------

resources = [f"bug_clip{i}" for i in range(1, 5)]

DATA_DIR = "recorded_data"

# Number of frames used for light position smoothing
SMOOTHING_WINDOW = 5


# --------------------------------------------------
# Orientation interpolation
# --------------------------------------------------

def interpolate_orientation(theta):
    """
    Interpolate orientation while respecting the fact that
    theta is defined modulo 180 degrees.

    We double the angle so that a 180-degree-periodic axis
    becomes a 360-degree-periodic direction, unwrap it,
    interpolate it, then divide by 2 again.
    """

    theta = theta.copy()

    # Work only with the known orientations first
    known = theta.notna()

    if known.sum() == 0:
        return theta

    # Convert known angles to radians after doubling
    doubled_rad = np.deg2rad(
        2 * theta.loc[known].to_numpy()
    )

    # Remove wrap discontinuities
    unwrapped = np.unwrap(doubled_rad)

    # Put unwrapped values back into a full Series
    unwrapped_series = pd.Series(
        np.nan,
        index=theta.index,
        dtype=float
    )

    unwrapped_series.loc[known] = unwrapped

    # Linear interpolation across missing frames
    unwrapped_series = unwrapped_series.interpolate(
        method="linear",
        limit_direction="both"
    )

    # Convert back to degrees and undo doubling
    interpolated_theta = (
        np.rad2deg(unwrapped_series) / 2
    ) % 180

    return interpolated_theta


# --------------------------------------------------
# Reconstruct one trajectory
# --------------------------------------------------

def reconstruct_trajectory(df):

    # Keep original detection flags unchanged
    detected_original = df["detected"].copy()

    # --------------------------------------------------
    # 1. Interpolate x and y
    # --------------------------------------------------

    df["x_cm"] = df["x_cm"].interpolate(
        method="linear",
        limit_direction="both"
    )

    df["y_cm"] = df["y_cm"].interpolate(
        method="linear",
        limit_direction="both"
    )


    # --------------------------------------------------
    # 2. Interpolate orientation
    # --------------------------------------------------

    df["theta"] = interpolate_orientation(
        df["theta"]
    )


    # --------------------------------------------------
    # 3. Smooth reconstructed x and y trajectory
    # --------------------------------------------------

    df["x_cm"] = df["x_cm"].rolling(
        window=SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    ).mean()

    df["y_cm"] = df["y_cm"].rolling(
        window=SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    ).mean()


    # --------------------------------------------------
    # 4. Restore original detection flags
    # --------------------------------------------------

    df["detected"] = detected_original

    return df


# --------------------------------------------------
# Process each Task 1 CSV
# --------------------------------------------------

for resource in resources:

    input_path = os.path.join(
        DATA_DIR,
        f"{resource}_data_raw.csv"
    )

    output_path = os.path.join(
        DATA_DIR,
        f"{resource}_data_interpolated.csv"
    )

    print(f"Processing {input_path}...")

    # Load raw Task 1 trajectory
    df = pd.read_csv(input_path)

    # Reconstruct continuous trajectory
    df = reconstruct_trajectory(df)

    # Save Task 2 trajectory
    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved {output_path}")


print("Trajectory reconstruction complete.")