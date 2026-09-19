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


### Notes:

`pixel_cailbration.py`: Since the camera viewpoint changed a little for each video, we needed to record the arena bounds and store the px to cm conversion factor specific to that video. This file displays the first frame of each video and has you click the bottom left and then bottom right inside corners of the arena. It then does the calculations needed to store the necessary data into `./recorded_data/arena_bounds.csv`

`detection_orientation.py`: This is the work done for retrieving the raw data needed for the first task. there is no interpolation being done at this point. 

`trajectory_reconstruction.py`: This script performs linear interpolation and some light smoothing on the raw data collected in task 1 and saves the outputs to ./recorded_data

`task2_visualization.py`: This script produces the annotated vieos with the trailing path of the preceding fewseconds using the interpolated data. saves outputs to ./recorded_data/task2_outputs
