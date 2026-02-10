# TheScript (Rosco AI)

A computer vision pipeline for real-time soccer match analytics, built by Tanner Selio and Adrian Manchado to assist MSOE's men's soccer coach, Rob Harrington, with data-driven insights from game footage. Using only a single camera setup and no expensive hardware, Rosco AI brings professional-level sports analytics to a D3 soccer program — detecting players and the ball, tracking them across frames, mapping their positions onto a real-world field model, and computing per-player and per-team metrics.

## Demos

- [Annotated match video with player speed, distance, and possession overlays](https://www.youtube.com/watch?v=HKj34-95-PM)
- [2D bird's-eye field projection with real-time player positions](https://www.youtube.com/watch?v=loeXq97BnCw)
- [Full presentation: AI-Driven Soccer Analysis Using Computer Vision](https://www.youtube.com/watch?v=UKSFIYSSHr8&t=52s)

## What We Built

### Player Detection & Team Classification
**YOLO** (You Only Look Once) detects players in each frame by identifying pixel patterns through convolutional neural networks. Teams are differentiated by analyzing jersey colors — pixel color averaging and **K-Means clustering** group similar colors to separate teams. Outlier colors (e.g., bright yellow) enable automatic detection of referees.

### Player Tracking
YOLO operates on single frames without memory, so **SAM 2** (Segment Anything Model 2) is used to track players across frames. SAM 2 maintains a rolling memory of past frames to link player appearances and locations, assigning each player a unique ID for continuous tracking. When a player leaves and re-enters the frame, YOLO re-prompts SAM 2 to recover their identity. Occlusions (players overlapping) are detected via IOU overlap and handled by re-prompting SAM 2 on affected frames.

### Ball Detection & Tracking
YOLO also detects the ball, which is challenging due to its small size, high speed, and frequent occlusions. When the ball is not visible, a **Kalman filter** predicts its position based on velocity and trajectory, maintaining continuous ball tracking even through occluded frames.

### Field Keypoint Detection
We trained a **YOLOv8x-pose** model to detect 12 key landmarks on a soccer field (center circle edges, midfield line intersections, penalty-box D-line intersections, etc.). Training data was annotated across 6 iterative dataset versions on [Roboflow](https://universe.roboflow.com/manchado53/keypoint-msoe/dataset/6) using Label Studio and converted to COCO/YOLO format. We also built a custom **CNN baseline** in TensorFlow/Keras with dual outputs (keypoint visibility classification + coordinate regression) to compare approaches.

### White-Line Enhancement Preprocessing
Before running keypoint inference, each frame is converted to HSV color space and a binary mask isolates white pixels. Morphological dilation thickens the detected field lines, making them more robust to varying lighting and camera conditions.

### Homography & Coordinate Transformation
Predicted keypoints in camera-pixel space are matched to their known real-world positions on a FIFA-standard field (107.79 m x 63.74 m). A homography matrix is computed via `cv2.findHomography` and **exponentially smoothed** (alpha = 0.8) across frames to avoid jitter. This matrix transforms any 2D pixel coordinate (player, ball) into metric field coordinates.

### Player Analytics
Using the bounding boxes from YOLO + SAM 2 and the homography-transformed field coordinates, the pipeline computes:
- **Speed** (m/s) from consecutive transformed positions
- **Total distance** traveled per player
- **Off-ball runs** (high-speed movement without possession)
- **Team identification** via color clustering with per-player color assignments

### Ball Possession Analysis
A `PossessionTracker` divides the field into a 3x3 grid of zones. Each frame, the closest player to the ball (within a threshold radius) is credited with possession. The system outputs per-team possession percentages both overall and per zone.

### Zone Occupancy Tracking
Field is split into defensive, middle, and attacking thirds. The system tracks how many players from each team occupy each zone over time, providing tactical formation insights.

### Visualization & Output
The pipeline generates annotated video with:
- Color-coded player dots by team
- Real-time speed (m/s) and cumulative distance (m) overlays per player
- Ball position marker
- Team possession percentages displayed in an on-screen HUD
- Player heatmaps showing movement density

## Project Structure

```
TheScript/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                                # All Python source code
│   ├── main.py                         #   Primary video processing pipeline
│   ├── main_cnn.py                     #   CNN-based variant of the pipeline
│   ├── constants.py                    #   Model path configuration
│   ├── soccer_util_last.py             #   Core utilities: preprocessing, homography,
│   │                                   #     CNN model, field constants, evaluation
│   ├── keypoint_model.py               #   YOLO pose model wrapper
│   ├── grouping_teams.py               #   Player-to-team mapping and color management
│   ├── distance_speed_stimator.py      #   Speed/distance from transformed coordinates
│   ├── possesion_tracker.py            #   Ball possession tracking by field zone
│   ├── stimating_team_position.py      #   Defensive/middle/attacking zone occupancy
│   ├── heatmap.py                      #   Player movement heatmap generation
│   ├── video_creator.py                #   2D field-map and annotated video rendering
│   └── transforming_json_keypoints.py  #   Label Studio JSON -> COCO format converter
│
├── notebooks/                          # Jupyter notebooks
│   ├── FullHomography_03_21_2025.ipynb #   Homography exploration
│   └── uploading.ipynb                 #   Data upload/processing
│
├── scripts/                            # SLURM job submission scripts
│   ├── run_job.sh                      #   Inference on DGX H100
│   └── train_yolo_pose_slurm.sh        #   YOLOv8 training on V100s
│
├── weights/                            # Model weight files (gitignored)
│   ├── best.pt                         #   Fine-tuned YOLOv8 pose weights
│   ├── yolov8x-pose.pt                 #   YOLOv8 Extra Large pose base weights
│   ├── yolov8n-pose.pt                 #   YOLOv8 Nano pose weights
│   └── yolov8n.pt                      #   YOLOv8 Nano detection weights
│
├── data/                               # Training datasets (gitignored)
│   └── keypoint-msoe-{1..6}/          #   Iterative Roboflow datasets (CC BY 4.0)
│
├── assets/                             # Static resources
│   └── viets field.PNG                 #   Reference soccer field diagram
│
├── logs/                               # SLURM job output logs (gitignored)
└── runs/                               # YOLOv8 training outputs (gitignored)
```

## How It Works

```
Video Frame
    │
    ├─► YOLO detection ──► Player bounding boxes ──► SAM 2 tracking (with memory)
    │                                                     │
    ├─► YOLO detection ──► Ball position ──► Kalman filter (when occluded)
    │                                             │
    ├─► HSV white-line enhancement                │
    │       │                                     │
    │       ▼                                     │
    │   YOLOv8-pose inference                     │
    │       │                                     │
    │       ▼                                     │
    │   12 field keypoints + visibility           │
    │       │                                     │
    │       ▼                                     ▼
    │   Homography (smoothed) ◄──────── Player/ball positions
    │       │
    │       ▼
    │   Perspective transform to field coords
    │       │
    │       ├──► Speed / Distance per player
    │       ├──► Ball Possession by team & zone
    │       └──► Zone Occupancy by team & third
    │                     │
    ▼                     ▼
         Annotated output video
```

## Technologies

| Category | Tools |
|----------|-------|
| Object Detection & Pose | YOLOv8 (Ultralytics) |
| Player Tracking | SAM 2 (Segment Anything Model 2) |
| Ball Trajectory Prediction | Kalman Filter |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Augmentation | Albumentations |
| Annotation | Label Studio, Roboflow |
| Team Clustering | scikit-learn (K-Means) |
| Visualization | Matplotlib, Seaborn |
| Compute | MSOE Rosie supercomputer (SLURM, NVIDIA V100 / T4 / DGX H100) |

## Field Geometry

Standard FIFA field measurements used for the homography destination points:

| Measurement | Value |
|-------------|-------|
| Field length | 107.79 m |
| Field width | 63.74 m |
| Midfield circle radius | 9.15 m |
| Penalty box width | 40.3 m |
| Penalty box depth | 16.5 m |
| D-line distance from endline | 20.15 m |

12 landmarks are tracked: center circle (top, bottom, left, right), midfield line (top, bottom), left/right D-line centers, and left/right D-line top/bottom intersections.

## Training

YOLOv8x-pose was fine-tuned on our custom keypoint dataset:

```bash
# Submitted via SLURM on Rosie
sbatch scripts/train_yolo_pose_slurm.sh
```

Configuration: 100 epochs, batch size 16, image size 640x640, mosaic augmentation disabled, single GPU.

The dataset evolved through 6 versions with increasing annotation quality and coverage, hosted on [Roboflow](https://universe.roboflow.com/manchado53/keypoint-msoe/dataset/6) under CC BY 4.0.

## Running Inference

```bash
# Submit to Rosie cluster
sbatch scripts/run_job.sh
```

Or run directly:

```bash
python src/main.py
```

The analytics pipeline (`src/main.py`) expects:
- A video file (e.g., `30SecondsAurora.mp4`)
- Player bounding boxes from YOLO + SAM 2 (serialized as pickle)
- Ball positions from YOLO + Kalman filter (serialized as pickle)

## Notebooks

The Jupyter notebooks in `notebooks/` automatically add `src/` to their Python path so they can import the project modules. Open them from the repo root:

```bash
cd TheScript
jupyter notebook notebooks/FullHomography_03_21_2025.ipynb
```

- **FullHomography_03_21_2025.ipynb** — End-to-end exploration of keypoint detection, homography estimation, player tracking, possession analysis, zone occupancy, and video generation. Loads pre-computed player bounding boxes and ball positions from `/data/ai_club/SoccerStats2024/`.
- **uploading.ipynb** — Downloads keypoint datasets from Roboflow and runs basic model inference tests.

## Dependencies

```
opencv-python
numpy
tensorflow
scikit-learn
matplotlib
albumentations
seaborn
ultralytics
keras-tuner
pickle5
```

Install with:

```bash
pip install -r requirements.txt
```
