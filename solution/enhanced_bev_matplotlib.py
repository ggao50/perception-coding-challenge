"""
Enhanced Bird's Eye View with YOLO Detections using Matplotlib
Matches ego_trajectory.py visualization style
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, deque
import pandas as pd
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


class EnhancedBEVMatplotlib:
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.rgb_path = self.dataset_path / "rgb"
        self.xyz_path = self.dataset_path / "xyz"
        self.bbox_file = self.dataset_path / "bboxes_light.csv"

        # Load custom YOLO model
        model_path = 'runs/driving_detection/yolov8m_driving/weights/best.pt'
        if Path(model_path).exists():
            print(f"Loading custom YOLO model from {model_path}")
            self.yolo = YOLO(model_path)
        else:
            print("Using default YOLO model")
            self.yolo = YOLO('yolov8n.pt')

        # YOLO tracking configuration
        self.tracker_cfg = dict(
            tracker='bytetrack.yaml',
            persist=True,
            conf=0.1,
            iou=0.5
        )

        # Class names and colors (matplotlib format)
        self.class_names = ['golf_cart', 'green_traffic_light', 'pedestrian',
                           'red_traffic_light', 'traffic_barrier']

        self.class_colors = {
            'golf_cart': 'gold',
            'green_traffic_light': 'lime',
            'pedestrian': 'blue',
            'red_traffic_light': 'red',
            'traffic_barrier': 'magenta'
        }

        self.class_markers = {
            'golf_cart': 's',  # square
            'green_traffic_light': '^',  # triangle up
            'pedestrian': 'o',  # circle
            'red_traffic_light': 'v',  # triangle down
            'traffic_barrier': 'D'  # diamond
        }

        # Object tracking history
        self.object_tracks = defaultdict(list)
        self.ego_trajectory = []
        self.traffic_light_positions_cam = []
        self.all_detections = []

    def extract_3d_position(self, bbox, xyz_file, frame_idx):
        """Extract 3D position from bounding box and depth data."""
        if not xyz_file.exists():
            return None

        xyz_data = np.load(xyz_file)["xyz"][:, :, :3]

        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2
        cy_ground = int(y1 + 0.8 * (y2 - y1))

        # Sample patch
        patch_size = 5
        y_min = max(0, cy_ground - patch_size)
        y_max = min(xyz_data.shape[0], cy_ground + patch_size)
        x_min = max(0, cx - patch_size)
        x_max = min(xyz_data.shape[1], cx + patch_size)

        patch = xyz_data[y_min:y_max, x_min:x_max]

        # Filter valid points
        valid_mask = (patch[:,:,0] != 0) & np.isfinite(patch).all(axis=2)

        if not np.any(valid_mask):
            return None

        valid_points = patch[valid_mask]
        position = np.mean(valid_points, axis=0)

        if not np.all(np.isfinite(position)):
            return None

        return position[:3]

    def detect_objects_in_frame(self, img, xyz_file, frame_idx):
        """Detect all objects in frame using YOLO and extract 3D positions."""
        import cv2

        # Run YOLO tracking
        results = self.yolo.track(img, **self.tracker_cfg, verbose=False)

        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            # Get track IDs
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(boxes)

            for i in range(len(boxes)):
                tid = int(track_ids[i])
                if tid == -1:
                    continue

                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])

                if cls_id >= len(self.class_names):
                    continue

                class_name = self.class_names[cls_id]

                # Extract 3D position
                position_3d = self.extract_3d_position(bbox, xyz_file, frame_idx)

                if position_3d is not None:
                    detections.append({
                        'track_id': tid,
                        'class': class_name,
                        'confidence': confidence,
                        'position_3d': position_3d,
                        'frame': frame_idx
                    })

                    self.object_tracks[tid].append({
                        'frame': frame_idx,
                        'class': class_name,
                        'position': position_3d,
                        'confidence': confidence
                    })

        return detections

    def compute_ego_trajectory(self):
        """Compute ego trajectory from traffic light observations."""
        bbox_df = pd.read_csv(self.bbox_file)

        # Extract traffic light positions
        for idx, row in bbox_df.iterrows():
            if row['x1'] == 0 and row['y1'] == 0:
                continue

            xyz_file = self.xyz_path / f"depth{idx:06d}.npz"
            if not xyz_file.exists():
                continue

            tl_pos = self.extract_3d_position(
                [row['x1'], row['y1'], row['x2'], row['y2']],
                xyz_file, idx
            )

            if tl_pos is not None:
                self.traffic_light_positions_cam.append(tl_pos[:2])

        if len(self.traffic_light_positions_cam) < 2:
            print("Not enough traffic light observations")
            return

        self.traffic_light_positions_cam = np.array(self.traffic_light_positions_cam)

        # Compute ego trajectory
        initial_tl = self.traffic_light_positions_cam[0]
        ego_positions = []

        ego_positions.append(np.array([-initial_tl[0], -initial_tl[1]]))

        for t in range(1, len(self.traffic_light_positions_cam)):
            delta_tl = self.traffic_light_positions_cam[t] - self.traffic_light_positions_cam[t-1]
            ego_motion = -delta_tl
            ego_positions.append(ego_positions[-1] + ego_motion)

        self.ego_trajectory = np.array(ego_positions)

        # Smooth trajectory
        if len(self.ego_trajectory) > 5:
            window = min(31, len(self.ego_trajectory))
            if window % 2 == 0:
                window -= 1
            window = max(5, window)

            self.ego_trajectory[:, 0] = savgol_filter(self.ego_trajectory[:, 0], window, 3)
            self.ego_trajectory[:, 1] = savgol_filter(self.ego_trajectory[:, 1], window, 3)

        print(f"Computed ego trajectory with {len(self.ego_trajectory)} points")

    def process_dataset(self):
        """Process entire dataset and collect all detections."""
        import cv2

        # First compute ego trajectory
        print("Computing ego trajectory...")
        self.compute_ego_trajectory()

        if len(self.ego_trajectory) == 0:
            print("Failed to compute ego trajectory")
            return

        # Get all RGB images
        rgb_files = sorted(self.rgb_path.glob("*.png"))
        print(f"Processing {len(rgb_files)} frames...")

        # Process each frame
        for frame_idx, rgb_file in enumerate(rgb_files):
            if frame_idx % 30 == 0:
                print(f"Processing frame {frame_idx}/{len(rgb_files)}")

            # Load RGB image
            img = cv2.imread(str(rgb_file))

            # Load XYZ data
            xyz_file = self.xyz_path / f"depth{frame_idx:06d}.npz"

            # Detect objects
            detections = self.detect_objects_in_frame(img, xyz_file, frame_idx)

            # Transform to world coordinates and store
            if frame_idx < len(self.ego_trajectory):
                ego_pos = self.ego_trajectory[frame_idx]
                for det in detections:
                    # Transform to world frame
                    world_x = det['position_3d'][0] + ego_pos[0]
                    world_y = det['position_3d'][1] + ego_pos[1]
                    det['world_position'] = [world_x, world_y]
                    self.all_detections.append(det)

        print(f"\nDetection Statistics:")
        class_counts = defaultdict(int)
        for det in self.all_detections:
            class_counts[det['class']] += 1

        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} detections")

        print(f"  Total unique tracks: {len(self.object_tracks)}")

    def create_animation(self, save_path: str = "enhanced_bev_matplotlib.mp4"):
        """Create animated BEV matching ego_trajectory.py style."""

        fig, ax = plt.subplots(figsize=(10, 10))

        # Initialize plot elements
        ax.set_xlabel('X (meters) - Forward Direction', fontsize=12)
        ax.set_ylabel('Y (meters) - Lateral', fontsize=12)
        ax.set_title('Enhanced BEV with Object Detection\n(Bird\'s Eye View)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Set axis limits based on trajectory
        if len(self.ego_trajectory) > 0:
            x_min = min(self.ego_trajectory[:, 0].min(), -5) - 10
            x_max = max(self.ego_trajectory[:, 0].max(), 5) + 10
            y_min = min(self.ego_trajectory[:, 1].min(), -5) - 10
            y_max = max(self.ego_trajectory[:, 1].max(), 5) + 10
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        def update_frame(frame_num):
            ax.clear()

            # Reset plot settings
            ax.set_xlabel('X (meters) - Forward Direction', fontsize=12)
            ax.set_ylabel('Y (meters) - Lateral', fontsize=12)
            ax.set_title(f'Enhanced BEV with Object Detection (Frame {frame_num})\n(Bird\'s Eye View)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Reference lines
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            # Plot ego trajectory up to current frame
            if frame_num < len(self.ego_trajectory):
                # Full trajectory in light blue
                ax.plot(self.ego_trajectory[:frame_num+1, 0],
                       self.ego_trajectory[:frame_num+1, 1],
                       'b-', linewidth=2, alpha=0.5, label='Ego Trajectory')

                # Color-coded by time
                ax.scatter(self.ego_trajectory[:frame_num+1, 0],
                          self.ego_trajectory[:frame_num+1, 1],
                          c=np.arange(frame_num+1), cmap='viridis', s=20, alpha=0.7)

                # Current ego position
                ego_x, ego_y = self.ego_trajectory[frame_num]
                ax.scatter(ego_x, ego_y, color='red', s=150, marker='s',
                          label='Ego Vehicle', zorder=5, edgecolors='darkred', linewidth=2)

            # Mark start position
            if len(self.ego_trajectory) > 0:
                ax.scatter(self.ego_trajectory[0, 0], self.ego_trajectory[0, 1],
                          color='green', s=100, marker='o', label='Start', zorder=4)

            # Traffic light at origin
            ax.scatter(0, 0, color='yellow', s=200, marker='*',
                      label='Traffic Light (World Origin)', zorder=5,
                      edgecolors='black', linewidth=2)
            ax.text(0.5, -1, 'Traffic Light\nLocation', fontsize=10, ha='left', color='gray')

            # Plot detected objects for current frame
            frame_detections = [d for d in self.all_detections if d['frame'] == frame_num]

            for class_name in self.class_names:
                class_dets = [d for d in frame_detections if d['class'] == class_name]
                if class_dets:
                    # Traffic lights should be fixed at origin, not moving
                    if class_name in ['green_traffic_light', 'red_traffic_light']:
                        # Place traffic lights at fixed position near origin
                        x_coords = [0.0]  # Fixed at traffic light position
                        y_coords = [0.0]

                        # Show traffic light state with appropriate color
                        ax.scatter(x_coords, y_coords,
                                  color=self.class_colors[class_name],
                                  marker=self.class_markers[class_name],
                                  s=150, label=class_name.replace('_', ' ').title(),
                                  alpha=0.9, edgecolors='black', linewidth=2, zorder=6)
                    else:
                        # Other objects move normally
                        x_coords = [d['world_position'][0] for d in class_dets]
                        y_coords = [d['world_position'][1] for d in class_dets]

                        ax.scatter(x_coords, y_coords,
                                  color=self.class_colors[class_name],
                                  marker=self.class_markers[class_name],
                                  s=80, label=class_name.replace('_', ' ').title(),
                                  alpha=0.8, edgecolors='black', linewidth=1)

            # Plot object trails (only for moving objects)
            for track_id, track_history in self.object_tracks.items():
                # Skip trails for traffic lights since they're stationary
                if track_history and track_history[0]['class'] in ['green_traffic_light', 'red_traffic_light']:
                    continue

                track_points = []
                for hist in track_history:
                    if hist['frame'] <= frame_num and hist['frame'] < len(self.ego_trajectory):
                        ego_pos = self.ego_trajectory[hist['frame']]
                        world_x = hist['position'][0] + ego_pos[0]
                        world_y = hist['position'][1] + ego_pos[1]
                        track_points.append([world_x, world_y])

                if len(track_points) > 1:
                    track_points = np.array(track_points)
                    # Fade trail based on age
                    for i in range(len(track_points) - 1):
                        alpha = max(0.1, 0.5 * (i / len(track_points)))
                        ax.plot(track_points[i:i+2, 0], track_points[i:i+2, 1],
                               color='gray', alpha=alpha, linewidth=1, linestyle=':')

            ax.legend(loc='upper right', fontsize=10)

        # Create animation
        print("Creating animation...")
        num_frames = min(len(self.ego_trajectory), len(list(self.rgb_path.glob("*.png"))))
        anim = animation.FuncAnimation(fig, update_frame, frames=num_frames,
                                      interval=50, blit=False)

        # Save animation
        print(f"Saving animation to {save_path}...")
        writer = animation.FFMpegWriter(fps=20, bitrate=2000)
        anim.save(save_path, writer=writer)
        plt.close()
        print(f"Saved enhanced BEV animation to {save_path}")

    def plot_static_bev(self, save_path: str = "enhanced_bev_static.png"):
        """Create static BEV plot with all detections overlaid."""

        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot complete ego trajectory
        if len(self.ego_trajectory) > 0:
            ax.plot(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1],
                   'b-', linewidth=2, label='Ego Trajectory', alpha=0.7)
            ax.scatter(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1],
                      c=np.arange(len(self.ego_trajectory)), cmap='viridis', s=20, alpha=0.5)

            # Mark start and end
            ax.scatter(self.ego_trajectory[0, 0], self.ego_trajectory[0, 1],
                      color='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(self.ego_trajectory[-1, 0], self.ego_trajectory[-1, 1],
                      color='red', s=100, marker='s', label='End', zorder=5)

        # Traffic light at origin
        ax.scatter(0, 0, color='yellow', s=200, marker='*',
                  label='Traffic Light (World Origin)', zorder=5,
                  edgecolors='black', linewidth=2)

        # Plot all detected objects with transparency
        for class_name in self.class_names:
            class_dets = [d for d in self.all_detections if d['class'] == class_name]
            if class_dets:
                x_coords = [d['world_position'][0] for d in class_dets]
                y_coords = [d['world_position'][1] for d in class_dets]

                ax.scatter(x_coords, y_coords,
                          color=self.class_colors[class_name],
                          marker=self.class_markers[class_name],
                          s=30, label=class_name.replace('_', ' ').title(),
                          alpha=0.3, edgecolors='none')

        # Formatting
        ax.set_xlabel('X (meters) - Forward Direction', fontsize=12)
        ax.set_ylabel('Y (meters) - Lateral', fontsize=12)
        ax.set_title('Complete Enhanced BEV with All Detections\n(Bird\'s Eye View)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.axis('equal')

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.text(0.5, -1, 'Traffic Light\nLocation', fontsize=10, ha='left', color='gray')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved static BEV plot to {save_path}")
        plt.close()


if __name__ == "__main__":
    generator = EnhancedBEVMatplotlib()
    generator.process_dataset()
    generator.create_animation()
    generator.plot_static_bev()