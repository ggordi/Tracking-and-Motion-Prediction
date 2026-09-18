import cv2
import numpy as np

MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
bg_subtractor=MOG2_subtractor
camera = cv2.VideoCapture("resources/bug_clip1.mp4")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # 1. Background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # 2. Remove MOG2 shadow pixels
    # MOG2 usually labels:
    # 0   = background
    # 127 = shadow
    # 255 = foreground
    _, threshold = cv2.threshold(
        fg_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # 3. Clean up small noisy regions
    cleaned = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    # 4. Fill small gaps inside the robot region
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 5. Find foreground contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 6. Remove tiny contours
    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > 100
    ]

    # 7. Keep ONLY the largest valid contour
    if valid_contours:

        robot_contour = max(
            valid_contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(robot_contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Cleaned Mask", cleaned)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()