"""
Task 2 Evaluation Script

Evaluates trajectory reconstruction on the held-out evaluation clips
(bug_clip5 and bug_clip6).

INPUT FILES:
    recorded_data/manual_task1_annotations.csv
    recorded_data/bug_clip5_data_interpolated.csv
    recorded_data/bug_clip6_data_interpolated.csv

RUN:
    python task2_evaluation.py

OUTPUT:
    1. Prints the Task 2 evaluation metrics needed for the report.
    2. Saves the manually evaluated frames and their errors to:
           recorded_data/task2_evaluation_results.csv

REPORT METRICS:
    - Fraction / percentage of frames that were interpolated
    - Longest consecutive interpolated gap
    - Mean moving-frame position error
    - Median moving-frame position error
    - 90th-percentile moving-frame position error

The original "detected" flag is preserved by the trajectory
reconstruction script:
    detected = 1 -> original Task 1 detection
    detected = 0 -> position was filled/reconstructed by Task 2
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
    "task2_evaluation_results.csv"
)

EVALUATION_CLIPS = [
    "bug_clip5",
    "bug_clip6",
]


# ============================================================
# HELPER FUNCTION: LONGEST INTERPOLATED GAP
# ============================================================

def longest_interpolated_gap(detected_column):
    """
    Returns the longest consecutive run of detected == 0.

    Example:
        detected = [1, 1, 0, 0, 0, 1, 0, 1]

    Longest interpolated gap = 3 frames.
    """

    longest = 0
    current = 0

    for detected in detected_column:

        if detected == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


# ============================================================
# LOAD MANUAL GROUND TRUTH
# ============================================================

annotations = pd.read_csv(ANNOTATION_PATH)

print(f"Loaded {len(annotations)} manual annotations.")


# ============================================================
# LOAD TASK 2 INTERPOLATED TRAJECTORIES
# ============================================================

trajectory_data = []

for clip in EVALUATION_CLIPS:

    path = os.path.join(
        DATA_DIR,
        f"{clip}_data_interpolated.csv"
    )

    df = pd.read_csv(path)

    # Add video name so that it matches the annotation CSV.
    df["video"] = f"{clip}.mp4"

    trajectory_data.append(df)


trajectory_data = pd.concat(
    trajectory_data,
    ignore_index=True
)


# ============================================================
# INTERPOLATION STATISTICS
# ============================================================

total_frames = len(trajectory_data)

interpolated_frames = int(
    (trajectory_data["detected"] == 0).sum()
)

direct_frames = int(
    (trajectory_data["detected"] == 1).sum()
)

interpolated_fraction = (
    interpolated_frames / total_frames
)

interpolated_percentage = (
    interpolated_fraction * 100
)


# ============================================================
# PER-CLIP INTERPOLATION STATISTICS
# ============================================================

clip_results = []

for clip in EVALUATION_CLIPS:

    video_name = f"{clip}.mp4"

    clip_data = trajectory_data[
        trajectory_data["video"] == video_name
    ].copy()

    # Sort by frame so consecutive gaps are measured correctly.
    clip_data = clip_data.sort_values("frame")

    clip_total = len(clip_data)

    clip_interpolated = int(
        (clip_data["detected"] == 0).sum()
    )

    clip_fraction = (
        clip_interpolated / clip_total
    )

    clip_percentage = (
        clip_fraction * 100
    )

    clip_longest_gap = longest_interpolated_gap(
        clip_data["detected"]
    )

    clip_results.append({
        "video": video_name,
        "total_frames": clip_total,
        "interpolated_frames": clip_interpolated,
        "interpolated_fraction": clip_fraction,
        "interpolated_percentage": clip_percentage,
        "longest_gap": clip_longest_gap,
    })


# Overall longest gap should NOT be calculated by simply concatenating
# the two videos, since the end of one video and beginning of another
# are not temporally connected.
overall_longest_gap = max(
    row["longest_gap"]
    for row in clip_results
)


# ============================================================
# MATCH MANUAL GROUND TRUTH TO TASK 2 TRAJECTORY
# ============================================================

results = annotations.merge(
    trajectory_data,
    on=["video", "frame"],
    how="left"
)


# Make sure all 30 manually annotated frames were matched.
missing = results["x_cm"].isna() | results["y_cm"].isna()

if missing.any():

    print("\nWARNING:")
    print(
        f"{missing.sum()} manually annotated frames do not have "
        "a reconstructed Task 2 position."
    )

    print(
        results.loc[
            missing,
            ["video", "frame"]
        ]
    )


# ============================================================
# POSITION ERROR
# ============================================================

results["position_error_cm"] = np.sqrt(
    (results["x_cm"] - results["x_GT"]) ** 2
    +
    (results["y_cm"] - results["y_GT"]) ** 2
)


# Keep only frames for which a reconstructed position exists.
valid = results.dropna(
    subset=[
        "x_cm",
        "y_cm",
        "x_GT",
        "y_GT",
    ]
).copy()


# ============================================================
# POSITION ACCURACY STATISTICS
# ============================================================

position_mean = valid[
    "position_error_cm"
].mean()

position_median = valid[
    "position_error_cm"
].median()

position_p90 = valid[
    "position_error_cm"
].quantile(0.90)


# ============================================================
# CHECK MANUAL FRAMES THAT WERE INTERPOLATED
# ============================================================

manual_interpolated = valid[
    valid["detected"] == 0
].copy()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TASK 2 EVALUATION RESULTS")
print("=" * 60)


# ------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------

print("\nINTERPOLATION")
print("-" * 60)

for row in clip_results:

    print(
        f"{row['video']}: "
        f"{row['interpolated_frames']} / "
        f"{row['total_frames']} frames interpolated "
        f"({row['interpolated_percentage']:.2f}%)"
    )

    print(
        f"    Longest interpolated gap: "
        f"{row['longest_gap']} frames"
    )


print("\nOverall:")
print(f"  Total frames:              {total_frames}")
print(f"  Direct detections:         {direct_frames}")
print(f"  Interpolated frames:       {interpolated_frames}")
print(f"  Fraction interpolated:     {interpolated_fraction:.4f}")
print(f"  Percentage interpolated:   {interpolated_percentage:.2f}%")
print(f"  Longest interpolated gap:  {overall_longest_gap} frames")


# ------------------------------------------------------------
# Manual reference set
# ------------------------------------------------------------

print("\nMANUAL REFERENCE SET")
print("-" * 60)

print(f"Annotated moving frames:     {len(results)}")
print(f"Frames with Task 2 position: {len(valid)}")
print(
    f"Of these, interpolated:      "
    f"{len(manual_interpolated)}"
)


# ------------------------------------------------------------
# Position accuracy
# ------------------------------------------------------------

print("\nMOVING-FRAME POSITION ERROR")
print("-" * 60)

print(f"Mean:                        {position_mean:.3f} cm")
print(f"Median:                      {position_median:.3f} cm")
print(f"90th percentile:             {position_p90:.3f} cm")


# ============================================================
# WORST POSITION ERRORS
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
            "detected",
            "position_error_cm",
        ]
    ].to_string(index=False)
)


# ============================================================
# SHOW MANUALLY ANNOTATED FRAMES THAT REQUIRED INTERPOLATION
# ============================================================

print("\nMANUALLY ANNOTATED FRAMES THAT WERE INTERPOLATED")
print("-" * 60)

if len(manual_interpolated) == 0:

    print("None.")

else:

    print(
        manual_interpolated[
            [
                "video",
                "frame",
                "position_error_cm",
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print(f"Saved per-frame results to {OUTPUT_PATH}")
print("=" * 60)