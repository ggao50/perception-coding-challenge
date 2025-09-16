"""
Smooth YOLOv8 Tracking with BYTETrack, EMA smoothing, and range-aware detection
Eliminates jitter and flickering with stable track IDs
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os
from collections import deque, defaultdict

class BoxEMA:
    """Exponential Moving Average for smooth box transitions"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.state = None  # (cx, cy, w, h)

    def update(self, xyxy):
        x1, y1, x2, y2 = map(float, xyxy)
        state_new = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
        if self.state is None:
            self.state = state_new
        else:
            self.state = self.alpha * state_new + (1 - self.alpha) * self.state
        return self

    def xyxy(self):
        cx, cy, w, h = self.state
        return (int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2))

class SmoothTracker:
    def __init__(self, dataset_folder="dataset"):
        # Load custom trained model
        custom_model_path = 'training_data/runs/driving_detection/yolov8m_driving/weights/best.pt'
        if Path(custom_model_path).exists():
            print(f"Loading custom trained YOLOv8 model from {custom_model_path}...")
            self.yolo = YOLO(custom_model_path)
        else:
            print("Custom model not found, using default YOLOv8n")
            self.yolo = YOLO('models/yolov8n.pt')

        # BYTETrack configuration for stable tracking
        self.tracker_cfg = dict(
            tracker='bytetrack.yaml',
            persist=True,
            conf=0.05,  # Very low for high recall
            iou=0.5
        )

        # Class configuration
        self.class_names = ['golf_cart', 'green_traffic_light', 'pedestrian', 'red_traffic_light', 'traffic_barrier']

        # Class-specific confidence thresholds
        self.class_thresholds = {
            'golf_cart': 0.50,
            'green_traffic_light': 0.10,
            'pedestrian': 0.25,
            'red_traffic_light': 0.10,
            'traffic_barrier': 0.30
        }

        # Colors for each class (BGR format)
        self.class_colors = {
            'golf_cart': (0, 255, 255),
            'green_traffic_light': (0, 255, 0),
            'pedestrian': (255, 0, 0),
            'red_traffic_light': (0, 0, 255),
            'traffic_barrier': (255, 0, 255)
        }

        # Per-track smoothing and history
        self.box_ema = {}  # track_id -> BoxEMA
        self.conf_hist = defaultdict(lambda: deque(maxlen=5))  # track_id -> confidence history
        self.class_hist = defaultdict(lambda: deque(maxlen=5))  # track_id -> class history
        self.appear_count = defaultdict(int)  # track_id -> frames appeared
        self.disappear_count = defaultdict(int)  # track_id -> frames disappeared
        self.active_tracks = {}  # track_id -> track info

        # Hysteresis thresholds
        self.appear_threshold = 2  # frames before showing
        self.disappear_threshold = 5  # frames before removing

        # Dataset paths
        self.dataset_folder = dataset_folder
        self.xyz_folder = os.path.join(dataset_folder, "xyz")

    def estimate_range(self, x1, y1, x2, y2, xyz_img):
        """Estimate object range using depth data from lower 25% of box"""
        try:
            # Sample lower 25% of box for ground contact
            y_start = int(y1 + 0.75 * (y2 - y1))
            y_end = min(y2, xyz_img.shape[0])
            x_start = max(0, x1)
            x_end = min(x2, xyz_img.shape[1])

            if y_start >= y_end or x_start >= x_end:
                return None

            # Extract depth points
            pts = xyz_img[y_start:y_end, x_start:x_end].reshape(-1, 3)

            # Filter valid points (non-zero, finite)
            valid = (pts[:, 0] > 0) & np.isfinite(pts).all(axis=1)

            if valid.any():
                # Return median X (forward distance)
                return np.median(pts[valid, 0])
        except:
            pass
        return None

    def get_range_adjusted_threshold(self, class_name, distance):
        """Adjust confidence threshold based on object distance"""
        base_threshold = self.class_thresholds[class_name]

        if distance is None:
            return base_threshold

        # Looser threshold for far objects (exponential decay)
        # More strict for very close objects
        if distance > 5:  # Far objects
            adjustment = 0.15 * np.exp(-(distance - 5) / 10)
            return max(0.08, base_threshold - adjustment)
        else:  # Close objects
            return base_threshold * 1.1  # Slightly stricter

    def process_frame(self, img, xyz_img=None, frame_idx=0):
        """Process single frame with smooth tracking"""

        # Run YOLO tracking with BYTETrack
        results = self.yolo.track(img, **self.tracker_cfg, verbose=False)

        if not results or results[0].boxes is None:
            # Decay disappear counts for missing tracks
            for tid in list(self.active_tracks.keys()):
                self.disappear_count[tid] += 1
                if self.disappear_count[tid] > self.disappear_threshold:
                    del self.active_tracks[tid]
            return img

        result = results[0]
        boxes = result.boxes

        # Get track IDs (critical for stability)
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(boxes)

        # Track which IDs are seen this frame
        seen_ids = set()

        # Process each detection
        for i in range(len(boxes)):
            tid = int(track_ids[i])
            if tid == -1:
                continue  # Skip untracked detections

            seen_ids.add(tid)

            # Get detection info
            xyxy = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])

            if cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                continue

            # Update or create box smoother
            if tid not in self.box_ema:
                self.box_ema[tid] = BoxEMA(alpha=0.35)
            self.box_ema[tid].update(xyxy)

            # Update confidence history
            self.conf_hist[tid].append(confidence)
            self.class_hist[tid].append(class_name)

            # Get smoothed box coordinates
            x1, y1, x2, y2 = self.box_ema[tid].xyxy()

            # Estimate range if XYZ available
            distance = None
            if xyz_img is not None:
                distance = self.estimate_range(x1, y1, x2, y2, xyz_img)

            # Get range-adjusted threshold
            threshold = self.get_range_adjusted_threshold(class_name, distance)

            # Check if track passes threshold (using windowed average)
            avg_conf = np.mean(self.conf_hist[tid])
            passes_threshold = avg_conf >= threshold * 0.9  # 90% of threshold for stability

            # Majority vote for class (handles class flicker)
            if len(self.class_hist[tid]) > 0:
                from collections import Counter
                class_name = Counter(self.class_hist[tid]).most_common(1)[0][0]

            # Update appear/disappear counts with hysteresis
            if passes_threshold:
                self.appear_count[tid] += 1
                self.disappear_count[tid] = 0

                # Add to active tracks if appeared enough
                if self.appear_count[tid] >= self.appear_threshold:
                    self.active_tracks[tid] = {
                        'box': (x1, y1, x2, y2),
                        'class': class_name,
                        'confidence': avg_conf,
                        'distance': distance,
                        'threshold': threshold
                    }
            else:
                self.disappear_count[tid] += 1
                self.appear_count[tid] = 0

        # Handle tracks that disappeared this frame
        for tid in list(self.active_tracks.keys()):
            if tid not in seen_ids:
                self.disappear_count[tid] += 1
                if self.disappear_count[tid] > self.disappear_threshold:
                    # Remove track after enough missing frames
                    del self.active_tracks[tid]
                    if tid in self.box_ema:
                        del self.box_ema[tid]

        # Draw all active tracks
        for tid, track_info in self.active_tracks.items():
            x1, y1, x2, y2 = track_info['box']
            class_name = track_info['class']
            confidence = track_info['confidence']
            distance = track_info['distance']

            # Get color
            color = self.class_colors.get(class_name, (255, 255, 255))

            # Draw smooth bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"ID{tid}: {class_name} {confidence:.2f}"
            if distance is not None:
                label += f" [{distance:.1f}m]"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(img, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return img

    def process_video(self, output_video="smooth_tracking.mp4"):
        """Process all frames with smooth tracking"""

        # Get image files
        rgb_folder = os.path.join(self.dataset_folder, "rgb")
        image_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])

        if not image_files:
            print(f"No PNG files found in {rgb_folder}")
            return

        print(f"Found {len(image_files)} images to process")
        print("\nUsing BYTETrack with:")
        print("  - EMA box smoothing (α=0.35)")
        print("  - Confidence windowing (5 frames)")
        print("  - Hysteresis (2 frames to appear, 5 to disappear)")
        print("  - Range-aware thresholds")
        print()

        # Read first image for dimensions
        first_img = cv2.imread(os.path.join(rgb_folder, image_files[0]))
        height, width = first_img.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

        # Reset tracking state
        self.box_ema.clear()
        self.conf_hist.clear()
        self.class_hist.clear()
        self.appear_count.clear()
        self.disappear_count.clear()
        self.active_tracks.clear()

        # Process each frame
        for idx, img_file in enumerate(image_files):
            if idx % 30 == 0:
                print(f"Processing frame {idx}/{len(image_files)} - Active tracks: {len(self.active_tracks)}")

            # Read RGB image
            img_path = os.path.join(rgb_folder, img_file)
            img = cv2.imread(img_path)

            # Read XYZ if available
            xyz_path = os.path.join(self.xyz_folder, img_file.replace('.png', '.npy'))
            xyz_img = None
            if os.path.exists(xyz_path):
                try:
                    xyz_img = np.load(xyz_path)
                except:
                    pass

            # Process frame with smooth tracking
            img_with_tracking = self.process_frame(img.copy(), xyz_img, idx)

            # Write to video
            out.write(img_with_tracking)

        out.release()
        print(f"\nSaved smooth tracking video to {output_video}")
        print(f"Final active tracks: {len(self.active_tracks)}")


if __name__ == "__main__":
    tracker = SmoothTracker()
    tracker.process_video()