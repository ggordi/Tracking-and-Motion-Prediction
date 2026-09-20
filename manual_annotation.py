"""
Manual annotation tool for Task 1 evaluation.

This script is used to create ground-truth position and orientation
annotations for moving frames from the held-out evaluation clips
(bug_clip5 and bug_clip6).

HOW TO USE:

1. Run:
       python manual_annotation.py

2. The video will begin playing at the start of the evaluation interval.

VIDEO CONTROLS:
    SPACE = Pause / resume video
    A     = Move backward one frame (while paused)
    D     = Move forward one frame (while paused)
    S     = Select the current frame for manual annotation
    Q     = Finish the current video and move to the next one

ANNOTATION CONTROLS:
    After pressing S, click the following three points in order:

        1. Center of the robot
        2. One point along the robot's longitudinal body axis
        3. A second point along the robot's longitudinal body axis

    ENTER = Save the annotation
    R     = Reset the three points and annotate the frame again
    ESC   = Cancel the current annotation without saving

FRAME SELECTION:
    Select only frames where the robot is moving.
    Use frames from a variety of conditions, including:
        - Normal / straight movement
        - Turning
        - Wall interaction
        - Shadows / lighting changes
        - Motion blur

OUTPUT:
    Annotations are saved after every selected frame to:

        recorded_data/manual_task1_annotations.csv

    The output includes the exact video/frame number, manual center
    position, body-axis points, ground-truth position in centimeters,
    and ground-truth orientation in degrees.

    Existing annotations are loaded when the script is restarted, so
    annotation can be completed across multiple sessions.
"""

import cv2
import pandas as pd
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

VIDEO_DIR = "resources"
CALIBRATION_PATH = "recorded_data/arena_bounds.csv"
OUTPUT_PATH = "recorded_data/manual_task1_annotations.csv"

VIDEOS = [
    "bug_clip5.mp4",
    "bug_clip6.mp4",
]

# Your detector evaluates seconds 10-80, so restrict manual
# annotations to the same part of the video.
START_TIME = 10
END_TIME = 80


# ============================================================
# LOAD CALIBRATION
# ============================================================

arena_bounds = pd.read_csv(CALIBRATION_PATH)

# Load existing annotations if you stop and restart.
if os.path.exists(OUTPUT_PATH):
    annotations = pd.read_csv(OUTPUT_PATH).to_dict("records")
    print(f"Loaded {len(annotations)} existing annotations.")
else:
    annotations = []


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_annotations():
    df = pd.DataFrame(annotations)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} annotations to {OUTPUT_PATH}")


# ============================================================
# MANUAL ANNOTATION
# ============================================================

def annotate_frame(frame, video_name, frame_number, fps, bounds):
    """
    User clicks:
        1. robot center
        2. one point along body axis
        3. another point along body axis
    """

    points = []
    display = frame.copy()

    instructions = [
        "Click ROBOT CENTER",
        "Click FIRST BODY-AXIS point",
        "Click SECOND BODY-AXIS point",
    ]

    window_name = "Manual Annotation"

    def mouse_callback(event, x, y, flags, param):
        nonlocal display

        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))

            # Draw clicked point
            cv2.circle(
                display,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            # Once body-axis points exist, draw axis
            if len(points) == 3:
                cv2.line(
                    display,
                    points[1],
                    points[2],
                    (0, 255, 0),
                    2
                )

            cv2.imshow(window_name, display)

            if len(points) < 3:
                print(instructions[len(points)])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n----------------------------------")
    print(f"Annotating {video_name}, frame {frame_number}")
    print(instructions[0])
    print("Press R to redo clicks.")
    print("Press ESC to cancel this annotation.")

    while True:

        # Rebuild display text
        text_display = display.copy()

        cv2.putText(
            text_display,
            f"Frame {frame_number} | {frame_number / fps:.2f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if len(points) < 3:
            instruction = instructions[len(points)]
        else:
            instruction = "Press ENTER to save | R = redo"

        cv2.putText(
            text_display,
            instruction,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(window_name, text_display)

        key = cv2.waitKey(20) & 0xFF

        # ESC = cancel
        if key == 27:
            cv2.destroyWindow(window_name)
            return None

        # R = reset clicks
        if key == ord("r"):
            points.clear()
            display = frame.copy()
            print("Clicks reset.")
            print(instructions[0])

        # ENTER = save once all three points exist
        if key in [10, 13] and len(points) == 3:
            break

    cv2.destroyWindow(window_name)

    # --------------------------------------------------------
    # Extract manual clicks
    # --------------------------------------------------------

    center_x_px, center_y_px = points[0]

    axis1_x, axis1_y = points[1]
    axis2_x, axis2_y = points[2]

    # --------------------------------------------------------
    # Convert center pixels -> arena centimeters
    # Same convention as detection_orientation.py
    # --------------------------------------------------------

    left_x = bounds["bottom_left_x"]
    bottom_y = bounds["bottom_left_y"]
    cm_per_pixel = bounds["cm_per_pixel"]

    x_gt = (center_x_px - left_x) * cm_per_pixel
    y_gt = (bottom_y - center_y_px) * cm_per_pixel

    # --------------------------------------------------------
    # Calculate body-axis orientation
    #
    # Image coordinates have y increasing DOWNWARD.
    # This matches the orientation convention used by the
    # existing OpenCV PCA detector.
    # --------------------------------------------------------

    dx = axis2_x - axis1_x
    dy = axis2_y - axis1_y

    theta_gt = np.degrees(
        np.arctan2(dy, dx)
    ) % 180

    return {
        "video": video_name,
        "frame": frame_number,
        "time_s": frame_number / fps,

        "center_x_px": center_x_px,
        "center_y_px": center_y_px,

        "axis1_x_px": axis1_x,
        "axis1_y_px": axis1_y,
        "axis2_x_px": axis2_x,
        "axis2_y_px": axis2_y,

        "x_GT": x_gt,
        "y_GT": y_gt,
        "theta_GT": theta_gt,
    }


# ============================================================
# PROCESS VIDEOS
# ============================================================

for video_name in VIDEOS:

    video_path = os.path.join(VIDEO_DIR, video_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    bounds_rows = arena_bounds[
        arena_bounds["video"] == video_name
    ]

    if bounds_rows.empty:
        print(f"ERROR: No calibration found for {video_name}")
        cap.release()
        continue

    bounds = bounds_rows.iloc[0]

    # Start at the same 10-second point as the detector.
    cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)

    paused = False

    print("\n==========================================")
    print(f"VIDEO: {video_name}")
    print("==========================================")
    print("SPACE = pause/play")
    print("A     = previous frame (while paused)")
    print("D     = next frame (while paused)")
    print("S     = select/annotate current frame")
    print("Q     = move to next video")
    print("==========================================")

    current_frame = None
    current_frame_number = None

    while True:

        if not paused or current_frame is None:

            ret, frame = cap.read()

            if not ret:
                break

            current_frame = frame

            current_frame_number = int(
                cap.get(cv2.CAP_PROP_POS_FRAMES)
            ) - 1

        current_time = current_frame_number / fps

        if current_time >= END_TIME:
            print("Reached end of evaluation interval.")
            break

        # Make display copy
        display = current_frame.copy()

        # Overlay useful information
        cv2.putText(
            display,
            f"{video_name} | Frame {current_frame_number} | "
            f"{current_time:.2f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        status = "PAUSED" if paused else "PLAYING"

        cv2.putText(
            display,
            f"{status} | SPACE pause | S annotate | Q next video",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Annotations so far: {len(annotations)}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.imshow("Frame Selection", display)

        # Play roughly at original frame rate.
        delay = 1 if not paused else 30
        key = cv2.waitKey(delay) & 0xFF

        # SPACE
        if key == 32:
            paused = not paused

        # Q
        elif key == ord("q"):
            break

        # A = previous frame
        elif key == ord("a") and paused:

            target = max(
                int(current_frame_number - 1),
                int(START_TIME * fps)
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, target)

            ret, frame = cap.read()

            if ret:
                current_frame = frame
                current_frame_number = target

        # D = next frame
        elif key == ord("d") and paused:

            target = current_frame_number + 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, target)

            ret, frame = cap.read()

            if ret:
                current_frame = frame
                current_frame_number = target

        # S = annotate
        elif key == ord("s"):

            paused = True

            # Prevent accidental duplicate frame annotations
            already_done = any(
                row["video"] == video_name
                and int(row["frame"]) == current_frame_number
                for row in annotations
            )

            if already_done:
                print(
                    f"Frame {current_frame_number} "
                    "has already been annotated."
                )
                continue

            result = annotate_frame(
                current_frame,
                video_name,
                current_frame_number,
                fps,
                bounds
            )

            if result is not None:

                annotations.append(result)

                # Save after EVERY annotation so work isn't lost
                save_annotations()

                print(
                    f"Saved annotation #{len(annotations)}: "
                    f"{video_name}, frame {current_frame_number}"
                )

    cap.release()


# ============================================================
# FINISH
# ============================================================

cv2.destroyAllWindows()
save_annotations()

print("\nDone!")
print(f"Total annotations: {len(annotations)}")