# Computer Vision Challenge: Ego-Trajectory & BEV Mapping

## Method

### Ego-Trajectory Estimation
We treat the traffic light as a fixed world reference point. By tracking its apparent motion in the camera frame, we can infer the inverse ego-vehicle motion. The 3D position of the traffic light is extracted from depth data using an 11x11 patch averaging technique to reduce noise. The trajectory is then smoothed using a combined approach: moving average filter followed by Savitzky-Golay filter and cubic spline interpolation for a physically plausible path.

### Multi-Object BEV Tracking
Object detection uses a custom-trained YOLOv8m model on 5 classes specific to this dataset. ByteTrack maintains consistent object IDs across frames. Each detected object's 3D position is extracted from the XYZ depth data and transformed to world coordinates relative to the traffic light origin. All trajectories undergo the same smoothing pipeline as the ego vehicle to ensure coherent motion patterns. Class-specific confidence thresholds were empirically tuned: barriers (0.8) to reduce false positives, pedestrians (0.62) to prevent backward walking artifacts, and golf cart/lights (0.1) for reliable detection.

## Assumptions

1. **Traffic light stationarity** - The traffic light remains fixed throughout the sequence, serving as our world origin
2. **Flat ground plane** - All objects move on a planar surface, allowing 2D trajectory projection
3. **Depth reliability** - XYZ depth data is accurate within ±10%, handled through patch averaging
4. **Consistent camera calibration** - Intrinsic parameters remain constant across all frames
5. **Object persistence** - Tracked objects maintain consistent visual appearance for ByteTrack

## Results

The ego vehicle travels approximately 100m, starting 35m behind the traffic light and stopping 8m before it. The trajectory shows smooth deceleration approaching the intersection. Eight objects were successfully tracked: one golf cart maintaining consistent forward motion, two pedestrians crossing the scene, multiple traffic barriers as static landmarks, and traffic lights with correct state changes (red/green). All trajectories exhibit smooth, physically plausible motion after filtering. Processing achieves 1.38 FPS on RTX 5070 Ti with full trajectory smoothing enabled.

**Output Files:**
- `trajectory.png/mp4` - Ego-only trajectory in world frame
- `enhanced_bev_smooth.png/mp4` - Complete BEV with all tracked objects
- `smooth_tracking.mp4` - RGB video with stabilized detections