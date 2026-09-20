"""
Task 1 Evaluation Script

Compares the Task 1 detector output against the manually annotated
ground-truth frames from the held-out evaluation clips.

INPUT FILES:
    recorded_data/manual_task1_annotations.csv
    recorded_data/bug_clip5_data_raw.csv
    recorded_data/bug_clip6_data_raw.csv

RUN:
    python task1_evaluation.py

OUTPUT:
    1. Prints Task 1 evaluation results to the terminal.
    2. Saves per-frame errors to:
           recorded_data/task1_evaluation_results.csv

REPORT METRICS:
    Detection:
        - Total number of evaluation frames
        - Number and percentage successfully detected

    Position error:
        - Mean
        - Median
        - 90th percentile
        - Units: centimeters

    Orientation error:
        - Mean
        - Median
        - 90th percentile
        - Units: degrees

Position and orientation accuracy are calculated only for manually
annotated frames where the Task 1 detector successfully detected
the robot.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "recorded_data"

ANNOTATION_PATH = os.path.join(
    DATA_DIR,
    "manual_task1_annotations.csv"
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "task1_evaluation_results.csv"
)

EVALUATION_CLIPS = [
    "bug_clip5",
    "bug_clip6",
]


# ============================================================
# LOAD MANUAL GROUND TRUTH
# ============================================================

annotations = pd.read_csv(ANNOTATION_PATH)

print(f"Loaded {len(annotations)} manual annotations.")


# ============================================================
# LOAD DETECTOR OUTPUTS
# ============================================================

detector_data = []

for clip in EVALUATION_CLIPS:

    path = os.path.join(
        DATA_DIR,
        f"{clip}_data_raw.csv"
    )

    df = pd.read_csv(path)

    # Add video name so we can match annotations to detector output.
    df["video"] = f"{clip}.mp4"

    detector_data.append(df)

detector_data = pd.concat(
    detector_data,
    ignore_index=True
)


# ============================================================
# MATCH MANUAL ANNOTATIONS TO DETECTOR FRAMES
# ============================================================

results = annotations.merge(
    detector_data,
    on=["video", "frame"],
    how="left"
)


# Check that every manually annotated frame was found
missing = results["detected"].isna()

if missing.any():

    print("\nWARNING:")
    print(
        f"{missing.sum()} manually annotated frames could not "
        "be matched to detector output."
    )

    print(
        results.loc[
            missing,
            ["video", "frame"]
        ]
    )


# ============================================================
# CALCULATE POSITION ERROR
# ============================================================

results["position_error_cm"] = np.sqrt(
    (results["x_cm"] - results["x_GT"]) ** 2
    +
    (results["y_cm"] - results["y_GT"]) ** 2
)


# ============================================================
# CALCULATE ORIENTATION ERROR
# ============================================================

# Raw difference between predicted and ground-truth orientation
angle_difference = np.abs(
    results["theta"] - results["theta_GT"]
)

# Since body orientation is defined modulo 180 degrees,
# theta and theta + 180 represent the same body axis.
results["orientation_error_deg"] = np.minimum(
    angle_difference,
    180 - angle_difference
)


# ============================================================
# VALID DETECTIONS FOR POSITION / ORIENTATION ACCURACY
# ============================================================

valid = results[
    results["detected"] == 1
].copy()

# Extra safety: exclude any rows where an estimate is missing.
valid = valid.dropna(
    subset=[
        "x_cm",
        "y_cm",
        "theta",
        "x_GT",
        "y_GT",
        "theta_GT",
    ]
)


# ============================================================
# CALCULATE REQUIRED STATISTICS
# ============================================================

position_mean = valid["position_error_cm"].mean()
position_median = valid["position_error_cm"].median()
position_p90 = valid["position_error_cm"].quantile(0.90)

orientation_mean = valid["orientation_error_deg"].mean()
orientation_median = valid["orientation_error_deg"].median()
orientation_p90 = valid["orientation_error_deg"].quantile(0.90)


# ============================================================
# DETECTION RATE OVER FULL EVALUATION CLIPS
# ============================================================

total_frames = len(detector_data)

detected_frames = int(
    detector_data["detected"].sum()
)

failed_frames = total_frames - detected_frames

detection_rate = (
    detected_frames / total_frames * 100
)


# Per-clip detection rates
clip_detection_results = []

for clip in EVALUATION_CLIPS:

    video_name = f"{clip}.mp4"

    clip_data = detector_data[
        detector_data["video"] == video_name
    ]

    clip_total = len(clip_data)

    clip_detected = int(
        clip_data["detected"].sum()
    )

    clip_failed = clip_total - clip_detected

    clip_rate = (
        clip_detected / clip_total * 100
    )

    clip_detection_results.append({
        "video": video_name,
        "total": clip_total,
        "detected": clip_detected,
        "failed": clip_failed,
        "rate": clip_rate,
    })


# ============================================================
# PRINT REPORT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TASK 1 EVALUATION RESULTS")
print("=" * 60)


# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

print("\nDETECTION")
print("-" * 60)

for row in clip_detection_results:

    print(
        f"{row['video']}: "
        f"{row['detected']} / {row['total']} frames "
        f"({row['rate']:.2f}%)"
    )

print("\nOverall:")
print(f"  Total frames:       {total_frames}")
print(f"  Detected frames:    {detected_frames}")
print(f"  Failed frames:      {failed_frames}")
print(f"  Detection rate:     {detection_rate:.2f}%")


# ------------------------------------------------------------
# Manual evaluation set
# ------------------------------------------------------------

print("\nMANUAL REFERENCE SET")
print("-" * 60)

print(f"Annotated frames:     {len(results)}")
print(f"Valid detections:     {len(valid)}")
print(
    f"Failed detections:    "
    f"{len(results) - len(valid)}"
)


# ------------------------------------------------------------
# Position
# ------------------------------------------------------------

print("\nPOSITION ERROR")
print("-" * 60)

print(f"Mean:                 {position_mean:.3f} cm")
print(f"Median:               {position_median:.3f} cm")
print(f"90th percentile:      {position_p90:.3f} cm")


# ------------------------------------------------------------
# Orientation
# ------------------------------------------------------------

print("\nORIENTATION ERROR")
print("-" * 60)

print(f"Mean:                 {orientation_mean:.3f} degrees")
print(f"Median:               {orientation_median:.3f} degrees")
print(f"90th percentile:      {orientation_p90:.3f} degrees")


# ============================================================
# IDENTIFY WORST FRAMES
# ============================================================

print("\nWORST POSITION ERRORS")
print("-" * 60)

worst_position = valid.nlargest(
    5,
    "position_error_cm"
)

print(
    worst_position[
        [
            "video",
            "frame",
            "position_error_cm"
        ]
    ].to_string(index=False)
)


print("\nWORST ORIENTATION ERRORS")
print("-" * 60)

worst_orientation = valid.nlargest(
    5,
    "orientation_error_deg"
)

print(
    worst_orientation[
        [
            "video",
            "frame",
            "orientation_error_deg"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE PER-FRAME RESULTS
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print(f"Saved per-frame results to {OUTPUT_PATH}")
print("=" * 60)