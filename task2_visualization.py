import cv2
import pandas as pd
import numpy as np
import os


# --------------------------------------------------
# Setup
# --------------------------------------------------

resources = [f"bug_clip{i}" for i in range(1, 5)]

VIDEO_DIR = "resources"
DATA_DIR = "recorded_data"
OUTPUT_DIR = "recorded_data/task2_outputs"

# How many seconds of trajectory to show behind the robot
TRAIL_SECONDS = 3

# Same time ranges used when generating the Task 1 data
video_times = {
    f"{resource}.mp4": {
        "start": 10,
        "end": 80
    }
    for resource in resources
}

# Load arena calibration
arena_bounds = pd.read_csv(
    os.path.join(DATA_DIR, "arena_bounds.csv")
)


# --------------------------------------------------
# Convert arena coordinates back to image pixels
# --------------------------------------------------

def cm_to_pixels(x_cm, y_cm, left_x, bottom_y, cm_per_pixel):

    x_px = int(
        x_cm / cm_per_pixel + left_x
    )

    y_px = int(
        bottom_y - y_cm / cm_per_pixel
    )

    return x_px, y_px


# --------------------------------------------------
# Process each video
# --------------------------------------------------

for video_name, times in video_times.items():

    resource = os.path.splitext(video_name)[0]

    print(f"Creating Task 2 video for {video_name}...")

    # --------------------------------------------------
    # Load reconstructed trajectory
    # --------------------------------------------------

    trajectory_path = os.path.join(
        DATA_DIR,
        f"{resource}_data_interpolated.csv"
    )

    trajectory = pd.read_csv(trajectory_path)


    # --------------------------------------------------
    # Load calibration for this video
    # --------------------------------------------------

    bounds = arena_bounds[
        arena_bounds["video"] == video_name
    ].iloc[0]

    left_x = bounds["bottom_left_x"]
    bottom_y = bounds["bottom_left_y"]
    cm_per_pixel = bounds["cm_per_pixel"]


    # --------------------------------------------------
    # Open original video
    # --------------------------------------------------

    video_path = os.path.join(
        VIDEO_DIR,
        video_name
    )

    camera = cv2.VideoCapture(video_path)

    fps = camera.get(cv2.CAP_PROP_FPS)

    frame_width = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    # Number of previous frames to show
    trail_frames = int(
        fps * TRAIL_SECONDS
    )


    # --------------------------------------------------
    # Create output video
    # --------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{resource}_task2_trajectory.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )


    # --------------------------------------------------
    # Jump to beginning of analyzed section
    # --------------------------------------------------

    camera.set(
        cv2.CAP_PROP_POS_MSEC,
        times["start"] * 1000
    )


    # --------------------------------------------------
    # Process frames
    # --------------------------------------------------

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        current_time = (
            camera.get(cv2.CAP_PROP_POS_MSEC) / 1000
        )

        if current_time >= times["end"]:
            break


        # Actual frame number in original video
        frame_number = int(
            camera.get(cv2.CAP_PROP_POS_FRAMES)
        ) - 1


        # --------------------------------------------------
        # Get preceding few seconds of trajectory
        # --------------------------------------------------

        start_trail_frame = (
            frame_number - trail_frames
        )

        trail = trajectory[
            (trajectory["frame"] >= start_trail_frame)
            &
            (trajectory["frame"] <= frame_number)
        ]


        # --------------------------------------------------
        # Convert trajectory points back to pixels
        # --------------------------------------------------

        trail_points = []

        for _, row in trail.iterrows():

            # Skip if reconstruction somehow still contains NaN
            if (
                pd.isna(row["x_cm"])
                or
                pd.isna(row["y_cm"])
            ):
                continue

            x_px, y_px = cm_to_pixels(
                row["x_cm"],
                row["y_cm"],
                left_x,
                bottom_y,
                cm_per_pixel
            )

            trail_points.append(
                (x_px, y_px)
            )


        # --------------------------------------------------
        # Draw trailing trajectory
        # --------------------------------------------------

        if len(trail_points) >= 2:

            points = np.array(
                trail_points,
                dtype=np.int32
            )

            cv2.polylines(
                frame,
                [points],
                False,
                (0, 255, 0),
                3
            )


        # --------------------------------------------------
        # Draw current reconstructed position
        # --------------------------------------------------

        current_row = trajectory[
            trajectory["frame"] == frame_number
        ]

        if not current_row.empty:

            row = current_row.iloc[0]

            if (
                not pd.isna(row["x_cm"])
                and
                not pd.isna(row["y_cm"])
            ):

                x_px, y_px = cm_to_pixels(
                    row["x_cm"],
                    row["y_cm"],
                    left_x,
                    bottom_y,
                    cm_per_pixel
                )

                cv2.circle(
                    frame,
                    (x_px, y_px),
                    6,
                    (255, 0, 0),
                    -1
                )


        # --------------------------------------------------
        # Display and save frame
        # --------------------------------------------------

        cv2.imshow(
            "Task 2 Trajectory",
            frame
        )

        video_writer.write(frame)

        # ESC to stop
        if cv2.waitKey(30) & 0xFF == 27:
            break


    # --------------------------------------------------
    # Finish this video
    # --------------------------------------------------

    camera.release()
    video_writer.release()

    print(
        f"Saved Task 2 video to {output_path}"
    )


cv2.destroyAllWindows()