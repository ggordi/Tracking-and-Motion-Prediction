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

Task 2:

- reconstruct the robot trajectory from the raw Task 1 detections

- linearly interpolate missing x and y positions

- interpolate orientation while accounting for the 180 degree ambiguity of the body axis

- apply light smoothing to the reconstructed trajectory

- preserve the original detection flag so interpolated frames can still be identified

- generate annotated output videos showing the robot and its recent trajectory

- evaluate the reconstructed trajectory using manually annotated reference frames


Task 3:

- predict the robot's position 1 second into the future using the reconstructed trajectory from Task 2

- train a GRU-based motion prediction model

- use the previous 2 seconds (60 frames at 30 FPS) of trajectory data as input

- GRU input features:

    relative x position

    relative y position

    x velocity

    y velocity

    sin(2 * theta)

    cos(2 * theta)

- predict the x and y displacement of the robot 1 second (30 frames) into the future

- use clips 1-3 for training

- use clip 4 for validation

- keep clips 5 and 6 completely held out for final evaluation

- save the model checkpoint with the lowest validation loss

- manually annotate 40 future positions for evaluation:

    20 evenly distributed start times from clip 5

    20 evenly distributed start times from clip 6

- compare the GRU against two baselines:

    stationary prediction

    constant velocity prediction

- evaluate all methods against the same manually annotated t + 1 second positions

- report mean, median, and 90th percentile Euclidean prediction error

- generate a box plot comparing prediction errors across all 40 evaluation points

- generate an example visualization showing:

    preceding observed trajectory

    position at prediction time t

    GRU prediction at t + 1 second

    stationary prediction

    constant velocity prediction

    manually annotated ground truth at t + 1 second


### Notes:

`pixel_cailbration.py`: Since the camera viewpoint changed a little for each video, we needed to record the arena bounds and store the px to cm conversion factor specific to that video. This file displays the first frame of each video and has you click the bottom left and then bottom right inside corners of the arena. It then does the calculations needed to store the necessary data into `./recorded_data/arena_bounds.csv`

`detection_orientation.py`: This is the work done for retrieving the raw data needed for the first task. there is no interpolation being done at this point. 

`trajectory_reconstruction.py`: This script performs linear interpolation and some light smoothing on the raw data collected in task 1 and saves the outputs to ./recorded_data

`task2_visualization.py`: This script produces the annotated vieos with the trailing path of the preceding fewseconds using the interpolated data. saves outputs to ./recorded_data/task2_outputs

`task3_train_gru.py`: This script creates sliding trajectory sequences from the Task 2 outputs and trains the GRU motion prediction model. Clips 1-3 are used for training and clip 4 is used for validation. Each training example uses the previous 2 seconds of motion to predict the robot's displacement 1 second into the future. The checkpoint with the lowest validation loss is saved to `./recorded_data/task3_gru.pth`.

`task3_annotation.py`: This script generates evenly spaced evaluation start times across held-out clips 5 and 6. For each start time, it displays the video frame 1 second later and has you manually click the center of the robot. These manually annotated future positions are saved to `./recorded_data/manual_task3_annotations.csv` and are used as the ground truth for all Task 3 prediction methods.

`task3_evaluation.py`: This script evaluates the trained GRU, stationary baseline, and constant velocity baseline on the same 40 manually annotated evaluation points from clips 5 and 6. It computes the Euclidean prediction error in cm and reports the mean, median, and 90th percentile for each method. It also saves the full evaluation results and a box plot of the prediction errors to `./recorded_data`.

`task3_visualization.py`: This script selects a representative Task 3 evaluation example with GRU error closest to the median GRU error. It plots the preceding 2 seconds of observed motion, the position at time t, the GRU prediction, both baseline predictions, and the manually annotated ground-truth position 1 second later. The figure is saved to `./recorded_data/task3_prediction_example.png`.

### Task 3 Results:

Evaluation was performed on 40 manually annotated prediction points across held-out clips 5 and 6.

Stationary baseline:

- Mean error: 5.115 cm

- Median error: 4.088 cm

- 90th percentile error: 10.328 cm


Constant velocity baseline:

- Mean error: 12.455 cm

- Median error: 11.721 cm

- 90th percentile error: 20.436 cm


GRU:

- Mean error: 3.646 cm

- Median error: 2.195 cm

- 90th percentile error: 7.723 cm