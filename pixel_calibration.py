# script to capture the arena boundary data for each video

import cv2
import pandas as pd
import os

# Videos to calibrate
video_names = [
    "bug_clip1.mp4",
    "bug_clip2.mp4",
    "bug_clip3.mp4",
    "bug_clip4.mp4",
    "bug_clip5.mp4",
    "bug_clip6.mp4"
]

RESOURCE_DIR = "resources"
OUTPUT_PATH = "recorded_data/arena_bounds.csv"

ARENA_WIDTH_CM = 30.5


def calibrate_video(video_name):
    """
    Display the first frame of a video and allow the user to click:
        1. Bottom-left inside corner of arena
        2. Bottom-right inside corner of arena

    Returns the calibration information for the video.
    """

    video_path = os.path.join(RESOURCE_DIR, video_name)

    camera = cv2.VideoCapture(video_path)

    ret, frame = camera.read()

    if not ret:
        print(f"Could not read {video_name}")
        camera.release()
        return None

    points = []

    # Make a copy so we can draw clicked points without
    # modifying the original frame
    display_frame = frame.copy()

    window_name = f"Calibrate: {video_name}"

    def click_point(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:

            points.append((x, y))

            # Show the selected point
            cv2.circle(
                display_frame,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            cv2.imshow(window_name, display_frame)

            if len(points) == 1:
                print(f"{video_name} bottom-left: ({x}, {y})")
                print("Now click the bottom-right inside corner.")

            elif len(points) == 2:
                print(f"{video_name} bottom-right: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_point)

    print(f"\n--- {video_name} ---")
    print("Click the BOTTOM-LEFT inside corner.")

    cv2.imshow(window_name, display_frame)

    # Wait until two points have been selected
    while len(points) < 2:
        cv2.waitKey(1)

    bottom_left = points[0]
    bottom_right = points[1]

    left_x, left_y = bottom_left
    right_x, right_y = bottom_right

    # Arena width in pixels
    arena_width_px = right_x - left_x

    # Conversion factor
    cm_per_pixel = ARENA_WIDTH_CM / arena_width_px

    print(f"Arena width: {arena_width_px} px")
    print(f"Scale: {cm_per_pixel:.5f} cm/pixel")

    camera.release()
    cv2.destroyWindow(window_name)

    return {
        "video": video_name,
        "bottom_left_x": left_x,
        "bottom_left_y": left_y,
        "bottom_right_x": right_x,
        "bottom_right_y": right_y,
        "arena_width_px": arena_width_px,
        "cm_per_pixel": cm_per_pixel
    }


# ----------------------------
# Calibrate all videos
# ----------------------------

calibration_data = []

for video_name in video_names:

    result = calibrate_video(video_name)

    if result is not None:
        calibration_data.append(result)


# ----------------------------
# Export calibration CSV
# ----------------------------

df = pd.DataFrame(calibration_data)

os.makedirs("recorded_data", exist_ok=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved arena calibration data to {OUTPUT_PATH}")

cv2.destroyAllWindows()