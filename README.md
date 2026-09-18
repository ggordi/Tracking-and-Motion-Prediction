# Tracking-and-Motion-Prediction

### Environment Setup
1. python -m venv .venv
2. source .venv/bin/activate


### Tasks:

Task 1:
- robot tracking
- center estimation 
- orientation as a line segment along the estimated body axis (180, head/end not important)
- convert center coordinates to cm from pixels
- store for each frame: 
    frame
    x_cm
    y_cm
    theta_deg
    whether detection succeeded
- Generate an annotated output video showing:
    robot outline
    center
    estimated body-axis line