# Computer Vision Challenge: Ego-Trajectory & BEV Mapping

## Overview
Complete solution for ego-vehicle trajectory estimation and multi-object tracking in Bird's Eye View using monocular RGB+depth data from a golf cart platform.

## Method

### Part A: Ego-Trajectory Estimation (solution/ego_trajectory.py)
- **Approach**: Traffic light as fixed world reference point
- **3D Extraction**: Sample 11x11 patch around traffic light center in XYZ depth data
- **Motion Estimation**: Track traffic light movement, invert for ego motion
- **Smoothing**: Combined approach - moving average + Savitzky-Golay filter + spline interpolation
- **Output**: trajectory.png/mp4 showing 100m path

### Part B: Multi-Object Tracking & BEV Mapping

#### enhanced_bev_smooth.py - Main BEV Solution
- **Detection**: Custom YOLOv8m model trained on dataset-specific classes
  - Training: 5 classes (golf_cart, pedestrian, traffic_barrier, green/red lights)
  - Model path: `training_data/runs/driving_detection/yolov8m_driving/weights/best.pt`
- **Tracking**: ByteTrack for persistent IDs across frames
- **3D Projection**: Convert detections to world coordinates using XYZ depth
- **Trajectory Smoothing**: Same combined method as ego (applied to all objects)
- **Confidence Thresholds** (optimized through testing):
  - Traffic barriers: 0.8 (high to reduce false positives)
  - Pedestrians: 0.62 (prevents backward walking artifacts)
  - Golf cart/lights: 0.1 (reliable detection)
- **GPU Acceleration**: Runs on NVIDIA RTX 5070 Ti

#### smooth_tracking.py - Enhanced Detection Visualization
- **Real-time Enhancements**:
  - EMA box smoothing (α=0.35)
  - 5-frame confidence windowing
  - Hysteresis (2 frames appear, 5 disappear)
  - Range-aware threshold adjustment
- **Output**: RGB video with stable bounding boxes

#### adaptive_detection.py - Early Prototype
- Initial prototype for class-specific thresholds
- Basic temporal smoothing (confidence decay)
- No tracking (frame-by-frame detection only)
- Superseded by smooth_tracking.py

## Results

**Ego Trajectory**:
- Start: (-35.53, -15.28) meters
- End: (-8.22, 0.72) meters
- Total: ~100m traveled

**Object Tracking**:
- 8 unique objects tracked with smooth paths
- Stable IDs maintained throughout sequence
- Performance: 1.38 FPS with full processing

**Output Files**:
- `enhanced_bev_smooth.mp4` - Complete animated BEV with all trajectories
- `enhanced_bev_smooth.png` - Static BEV visualization
- `smooth_tracking.mp4` - RGB video with enhanced tracking overlays
- `adaptive_detections.mp4` - Basic detection visualization
- `trajectory.mp4/png` - Ego-only trajectory

## Key Assumptions
1. Traffic light remains stationary (world reference)
2. Flat ground plane approximation
3. Constant camera intrinsics
4. Depth accuracy ±10% tolerance
5. Objects maintain consistent appearance

## Usage
```bash
# Main BEV with all objects
python enhanced_bev_smooth.py

# Enhanced tracking visualization
python smooth_tracking.py

# Basic detection (no tracking)
python adaptive_detection.py

# Ego trajectory only (in solution/)
python solution/ego_trajectory.py
```

**Dependencies**: numpy, pandas, matplotlib, opencv-python, ultralytics, scipy, torch