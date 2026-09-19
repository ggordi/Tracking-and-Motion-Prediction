import cv2
import numpy as np
import os
import pandas as pd


# --------------------------------------------------
# Setup
# --------------------------------------------------

resources = [f"bug_clip{i}" for i in range(1, 5)]

# Only clips 1-4 are used for development.
# Clips 5-6 remain held out for evaluation.
video_times = {
    f"{resource}.mp4": {
        "start": 10,
        "end": 80
    }
    for resource in resources
}

# Load arena calibration data
arena_bounds = pd.read_csv("recorded_data/arena_bounds.csv")

# Store raw frame results for each video
video_data = {
    resource: []
    for resource in resources
}

# Make sure output directory exists
os.makedirs("recorded_data", exist_ok=True)

# Kernel used for morphological cleanup
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


# --------------------------------------------------
# Process each video
# --------------------------------------------------

for video_name, times in video_times.items():

    start_time = times["start"]
    end_time = times["end"]

    resource = os.path.splitext(video_name)[0]

    print(f"Processing {video_name}...")

    # --------------------------------------------------
    # Get calibration data for this video
    # --------------------------------------------------

    bounds = arena_bounds[
        arena_bounds["video"] == video_name
    ].iloc[0]

    left_x = bounds["bottom_left_x"]
    bottom_y = bounds["bottom_left_y"]
    cm_per_pixel = bounds["cm_per_pixel"]


    # --------------------------------------------------
    # Open video
    # --------------------------------------------------

    camera = cv2.VideoCapture(
        os.path.join("resources", video_name)
    )

    # Get video properties for output video
    fps = camera.get(cv2.CAP_PROP_FPS)

    frame_width = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    # --------------------------------------------------
    # Create annotated video writer
    # --------------------------------------------------

    annotated_video_path = (
        f"recorded_data/task1_outputs/{resource}_task1_annotated.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        annotated_video_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )


    # --------------------------------------------------
    # New background model for each video
    # --------------------------------------------------

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        detectShadows=True
    )


    # --------------------------------------------------
    # Jump to this video's start time
    # --------------------------------------------------

    camera.set(
        cv2.CAP_PROP_POS_MSEC,
        start_time * 1000
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

        if current_time >= end_time:
            break

        # Actual frame number from original video
        frame_number = int(
            camera.get(cv2.CAP_PROP_POS_FRAMES)
        ) - 1


        # --------------------------------------------------
        # Default frame values
        # --------------------------------------------------
        # Assume detection fails unless successfully found.

        detected = 0
        x_cm = np.nan
        y_cm = np.nan
        theta = np.nan


        # --------------------------------------------------
        # 1. Background subtraction
        # --------------------------------------------------

        fg_mask = bg_subtractor.apply(frame)


        # --------------------------------------------------
        # 2. Remove MOG2 shadow pixels
        # --------------------------------------------------

        _, threshold = cv2.threshold(
            fg_mask,
            200,
            255,
            cv2.THRESH_BINARY
        )


        # --------------------------------------------------
        # 3. Remove small noise
        # --------------------------------------------------

        cleaned = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )


        # --------------------------------------------------
        # 4. Fill small gaps
        # --------------------------------------------------

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel
        )


        # --------------------------------------------------
        # 5. Find foreground contours
        # --------------------------------------------------

        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # --------------------------------------------------
        # 6. Remove tiny contours
        # --------------------------------------------------

        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) > 100
        ]


        # --------------------------------------------------
        # 7. Select robot
        # --------------------------------------------------

        if valid_contours:

            # Assume largest valid foreground object is robot
            robot_contour = max(
                valid_contours,
                key=cv2.contourArea
            )


            # --------------------------------------------------
            # Draw robot contour / outline
            # --------------------------------------------------

            cv2.drawContours(
                frame,
                [robot_contour],
                -1,
                (0, 255, 255),
                2
            )


            # --------------------------------------------------
            # 8. Estimate robot center
            # --------------------------------------------------

            M = cv2.moments(robot_contour)

            if M["m00"] != 0:

                cx = int(
                    M["m10"] / M["m00"]
                )

                cy = int(
                    M["m01"] / M["m00"]
                )


                # Draw center
                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    -1
                )


                # --------------------------------------------------
                # 9. Estimate orientation using PCA
                # --------------------------------------------------

                data_pts = (
                    robot_contour
                    .reshape(-1, 2)
                    .astype(np.float32)
                )

                mean, eigenvectors = cv2.PCACompute(
                    data_pts,
                    mean=None
                )

                # First principal component gives
                # dominant body-axis direction.
                vx, vy = eigenvectors[0]


                # Convert direction vector to angle.
                # Modulo 180 because front/back are equivalent.
                theta = (
                    np.degrees(
                        np.arctan2(vy, vx)
                    ) % 180
                )


                # --------------------------------------------------
                # 10. Draw estimated body axis
                # --------------------------------------------------

                line_length = 50

                pt1 = (
                    int(cx - vx * line_length),
                    int(cy - vy * line_length)
                )

                pt2 = (
                    int(cx + vx * line_length),
                    int(cy + vy * line_length)
                )

                cv2.line(
                    frame,
                    pt1,
                    pt2,
                    (0, 255, 0),
                    2
                )


                # --------------------------------------------------
                # 11. Convert pixel center to centimeters
                # --------------------------------------------------

                # x increases left -> right
                x_cm = (
                    cx - left_x
                ) * cm_per_pixel

                # OpenCV y increases downward.
                # Reverse it so arena y increases upward.
                y_cm = (
                    bottom_y - cy
                ) * cm_per_pixel


                # --------------------------------------------------
                # 12. Detection succeeded
                # --------------------------------------------------

                detected = 1


        # --------------------------------------------------
        # 13. Record this frame
        # --------------------------------------------------
        # Happens for every frame, including failed detections.

        video_data[resource].append({
            "frame": frame_number,
            "x_cm": x_cm,
            "y_cm": y_cm,
            "theta": theta,
            "detected": detected
        })


        # --------------------------------------------------
        # 14. Save annotated frame to output video
        # --------------------------------------------------

        video_writer.write(frame)


        # --------------------------------------------------
        # Display
        # --------------------------------------------------

        cv2.imshow(
            "Foreground Mask",
            fg_mask
        )

        cv2.imshow(
            "Cleaned Mask",
            cleaned
        )

        cv2.imshow(
            "Detection",
            frame
        )

        # Press ESC to stop
        if cv2.waitKey(30) & 0xFF == 27:
            break


    # --------------------------------------------------
    # Finish this video
    # --------------------------------------------------

    camera.release()
    video_writer.release()

    print(
        f"Saved annotated video to {annotated_video_path}"
    )


cv2.destroyAllWindows()


# --------------------------------------------------
# Export raw Task 1 data
# --------------------------------------------------

for resource, rows in video_data.items():

    df = pd.DataFrame(rows)

    output_path = (
        f"recorded_data/{resource}_data_raw.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved {len(df)} frames to {output_path}"
    )