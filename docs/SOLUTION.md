# Computer Vision Challenge: Ego-Trajectory & BEV Solution

## Quick Start

### Required Outputs (Part A)
- `trajectory.png` - Static BEV of ego vehicle path
- `trajectory.mp4` - Animated trajectory

### Enhanced Outputs (Part B)
- `enhanced_bev_static.png` - Complete scene with all objects
- `enhanced_bev_matplotlib.mp4` - Animated BEV with tracking

## File Structure

```
.
├── dataset/                  # Input data (RGB, XYZ, bboxes)
├── solution/                 # Core implementation
│   ├── ego_trajectory.py     # Part A solution
│   └── enhanced_bev_matplotlib.py  # Part B solution
├── outputs/                  # Generated visualizations
│   ├── videos/              # All MP4 outputs
│   └── images/              # All PNG outputs
├── experiments/             # Alternative implementations
├── training_data/           # YOLO training datasets
└── models/                  # Trained YOLO weights

```

## Running the Code

```bash
# Part A - Basic trajectory
python solution/ego_trajectory.py

# Part B - Enhanced BEV with detections
python solution/enhanced_bev_matplotlib.py
```

## Key Features

1. **Accurate ego trajectory** using traffic light as world reference
2. **Custom YOLO model** trained on dataset (5 classes)
3. **Multi-object tracking** (golf cart, barriers, pedestrians)
4. **Smooth visualization** with filtered trajectories
5. **3D depth integration** for accurate BEV positioning

## Technical Approach

- **Coordinate System**: Traffic light at origin, ground plane projection
- **Motion Estimation**: Inverse of traffic light apparent motion
- **Object Detection**: YOLOv8 with ByteTrack for stable IDs
- **Smoothing**: Savitzky-Golay filter + EMA for jitter reduction