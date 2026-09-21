"""
Task 3 Manual Prediction Annotation

For each evaluation start time t, displays the frame approximately
one second later (t + 1 s).

Click the robot CENTER on that future frame.

The frame is scaled down for display only. Click coordinates are
converted back to the original image coordinates before calibration.
"""

import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

VIDEO_DIR = "resources"
DATA_DIR = "recorded_data"

CALIBRATION_PATH = os.path.join(
    DATA_DIR,
    "arena_bounds.csv"
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "manual_task3_annotations.csv"
)

EVALUATION_CLIPS = [
    "bug_clip5",
    "bug_clip6",
]

START_TIME = 12
END_TIME = 78

POINTS_PER_CLIP = 20

# Scale used ONLY for displaying the annotation window.
# Try 0.5 if 0.65 is still too large.
DISPLAY_SCALE = 0.65


# ============================================================
# LOAD CALIBRATION
# ============================================================

arena_bounds = pd.read_csv(
    CALIBRATION_PATH
)


# ============================================================
# LOAD EXISTING ANNOTATIONS
# ============================================================

if os.path.exists(OUTPUT_PATH):

    annotations = pd.read_csv(
        OUTPUT_PATH
    ).to_dict("records")

    print(
        f"Loaded {len(annotations)} "
        f"existing annotations."
    )

else:

    annotations = []


def save_annotations():

    pd.DataFrame(
        annotations
    ).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved {len(annotations)} "
        f"annotations."
    )


# ============================================================
# ANNOTATE ONE FRAME
# ============================================================

def annotate_center(
    frame,
    video_name,
    start_frame,
    target_frame,
    bounds
):

    point = []

    # --------------------------------------------------------
    # Resize frame FOR DISPLAY ONLY
    # --------------------------------------------------------

    original_height, original_width = frame.shape[:2]

    display_width = int(
        original_width * DISPLAY_SCALE
    )

    display_height = int(
        original_height * DISPLAY_SCALE
    )

    base_display = cv2.resize(
        frame,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )

    display = base_display.copy()

    window_name = "Task 3 Ground Truth"

    # --------------------------------------------------------
    # Mouse callback
    # --------------------------------------------------------

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        nonlocal display

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(point) == 0
        ):

            # Store DISPLAY coordinates for now.
            point.append(
                (x, y)
            )

            cv2.circle(
                display,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    # --------------------------------------------------------
    # Annotation loop
    # --------------------------------------------------------

    while True:

        shown = display.copy()

        # Text sizes are slightly smaller for laptop display.
        cv2.putText(
            shown,
            f"Start: {start_frame}  Target: {target_frame}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            shown,
            "Click center | ENTER save | R redo | ESC skip",
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(
            window_name,
            shown
        )

        key = cv2.waitKey(20) & 0xFF

        # ESC = skip
        if key == 27:

            cv2.destroyWindow(
                window_name
            )

            return None

        # R = redo
        if key == ord("r"):

            point.clear()
            display = base_display.copy()

        # ENTER = save
        if (
            key in [10, 13]
            and len(point) == 1
        ):
            break

    cv2.destroyWindow(
        window_name
    )

    # --------------------------------------------------------
    # Convert display click -> ORIGINAL image coordinates
    # --------------------------------------------------------

    display_x, display_y = point[0]

    x_px = display_x / DISPLAY_SCALE
    y_px = display_y / DISPLAY_SCALE

    # --------------------------------------------------------
    # Pixel -> centimeter conversion
    # --------------------------------------------------------

    left_x = bounds[
        "bottom_left_x"
    ]

    bottom_y = bounds[
        "bottom_left_y"
    ]

    cm_per_pixel = bounds[
        "cm_per_pixel"
    ]

    x_gt = (
        x_px - left_x
    ) * cm_per_pixel

    y_gt = (
        bottom_y - y_px
    ) * cm_per_pixel

    return {
        "video": video_name,
        "start_frame": start_frame,
        "target_frame": target_frame,
        "x_GT": x_gt,
        "y_GT": y_gt,
    }


# ============================================================
# PROCESS EVALUATION CLIPS
# ============================================================

for clip in EVALUATION_CLIPS:

    video_name = f"{clip}.mp4"

    video_path = os.path.join(
        VIDEO_DIR,
        video_name
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            f"Could not open {video_path}"
        )

        continue

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    future_frames = int(
        round(fps)
    )

    bounds_rows = arena_bounds[
        arena_bounds["video"]
        == video_name
    ]

    if bounds_rows.empty:

        print(
            f"No calibration for "
            f"{video_name}"
        )

        cap.release()
        continue

    bounds = bounds_rows.iloc[0]

    # Generate evenly distributed evaluation times.
    start_times = np.linspace(
        START_TIME,
        END_TIME - 1,
        POINTS_PER_CLIP
    )

    for start_time in start_times:

        start_frame = int(
            round(start_time * fps)
        )

        target_frame = (
            start_frame
            + future_frames
        )

        # Don't annotate the same point twice.
        already_done = any(
            row["video"] == video_name
            and
            row["start_frame"] == start_frame
            for row in annotations
        )

        if already_done:
            continue

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame
        )

        ret, frame = cap.read()

        if not ret:
            continue

        result = annotate_center(
            frame,
            video_name,
            start_frame,
            target_frame,
            bounds
        )

        if result is not None:

            annotations.append(
                result
            )

            save_annotations()

    cap.release()


cv2.destroyAllWindows()

save_annotations()

print(
    "\nTask 3 annotation complete."
)