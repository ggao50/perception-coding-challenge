"""
Enhanced Bird's Eye View with Smooth Trajectories
Applies ego_trajectory.py smoothing to all tracked objects
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
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')


class EnhancedBEVSmooth:
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.rgb_path = self.dataset_path / "rgb"
        self.xyz_path = self.dataset_path / "xyz"
        self.bbox_file = self.dataset_path / "bboxes_light.csv"

        # Load custom YOLO model
        model_path = 'training_data/runs/driving_detection/yolov8m_driving/weights/best.pt'
        if Path(model_path).exists():
            print(f"Loading custom YOLO model from {model_path}")
            self.yolo = YOLO(model_path)
        else:
            print("Using default YOLO model")
            self.yolo = YOLO('models/yolov8n.pt')

        # Use GPU if available
        import torch
        if torch.cuda.is_available():
            self.yolo.to('cuda')
            print(f"  YOLO running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  YOLO running on CPU")

        # YOLO tracking configuration
        import torch
        self.tracker_cfg = dict(
            tracker='bytetrack.yaml',
            persist=True,
            conf=0.1,
            iou=0.5,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Class-specific confidence thresholds
        self.confidence_thresholds = {
            'traffic_barrier': 0.8,
            'golf_cart': 0.1,
            'green_traffic_light': 0.1,
            'red_traffic_light': 0.1,
            'pedestrian': 0.62  # Higher threshold to prevent false positives and backwards walking
        }

        # Class names and colors (matplotlib format)
        self.class_names = ['golf_cart', 'green_traffic_light', 'pedestrian',
                           'red_traffic_light', 'traffic_barrier']

        self.class_colors = {
            'golf_cart': 'gold',
            'green_traffic_light': 'lime',
            'pedestrian': 'cyan',
            'red_traffic_light': 'red',
            'traffic_barrier': 'orange'
        }

        self.class_markers = {
            'golf_cart': 's',
            'green_traffic_light': '^',
            'pedestrian': 'o',
            'red_traffic_light': 'v',
            'traffic_barrier': 'D'
        }

        # Storage for trajectories and detections
        self.ego_trajectory = []
        self.traffic_light_positions_cam = []
        self.object_tracks = defaultdict(list)
        self.smoothed_tracks = {}
        self.all_detections = []
        self.last_valid_tl_pos = None

    def smooth_trajectory(self, trajectory: np.ndarray, method: str = 'combined') -> np.ndarray:
        """
        Apply ego_trajectory.py smoothing method to any trajectory.
        This is the key function that creates smooth paths.
        """
        if len(trajectory) < 5:
            return trajectory

        smoothed = trajectory.copy()

        if method == 'combined':
            # Step 1: Apply moving average to remove high-frequency noise
            window = 11
            temp = trajectory.copy()
            for i in range(len(trajectory)):
                start = max(0, i - window // 2)
                end = min(len(trajectory), i + window // 2 + 1)
                temp[i] = np.mean(trajectory[start:end], axis=0)

            # Step 2: Apply Savitzky-Golay for smooth derivatives
            window_length = min(51, len(temp) if len(temp) % 2 == 1 else len(temp) - 1)
            window_length = max(5, window_length)

            smoothed[:, 0] = savgol_filter(temp[:, 0], window_length, 3)
            smoothed[:, 1] = savgol_filter(temp[:, 1], window_length, 3)

            # Step 3: For moving objects, ensure reasonable motion
            # (skip monotonic enforcement for objects that can move in any direction)

        elif method == 'savgol':
            # Simple Savitzky-Golay filter
            window_length = min(31, len(trajectory) if len(trajectory) % 2 == 1 else len(trajectory) - 1)
            window_length = max(5, window_length)

            smoothed[:, 0] = savgol_filter(trajectory[:, 0], window_length, 3)
            smoothed[:, 1] = savgol_filter(trajectory[:, 1], window_length, 3)

        elif method == 'spline':
            # Fit a spline to the trajectory
            t = np.arange(len(trajectory))
            # Use lower smoothing factor for more aggressive smoothing
            spl_x = UnivariateSpline(t, trajectory[:, 0], s=50)
            spl_y = UnivariateSpline(t, trajectory[:, 1], s=50)
            smoothed[:, 0] = spl_x(t)
            smoothed[:, 1] = spl_y(t)

        return smoothed

    def extract_3d_position(self, bbox, xyz_file, frame_idx):
        """Extract 3D position from depth data with patch averaging."""
        if not xyz_file.exists():
            return None

        x1, y1, x2, y2 = bbox
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        xyz_data = np.load(xyz_file)["xyz"]
        xyz = xyz_data[:, :, :3]

        # Use patch averaging for robustness
        patch_size = 5
        v_min = max(0, v - patch_size)
        v_max = min(xyz.shape[0], v + patch_size)
        u_min = max(0, u - patch_size)
        u_max = min(xyz.shape[1], u + patch_size)

        patch = xyz[v_min:v_max, u_min:u_max]

        # Filter out invalid points
        valid_mask = (patch[:,:,0] != 0) & np.isfinite(patch[:,:,0]) & \
                    np.isfinite(patch[:,:,1]) & np.isfinite(patch[:,:,2])

        if not np.any(valid_mask):
            center_point = xyz[v, u]
            if np.all(np.isfinite(center_point)) and center_point[0] != 0:
                return center_point
            return None

        # Average valid points
        valid_points = patch[valid_mask]
        position = np.mean(valid_points, axis=0)

        if not np.all(np.isfinite(position)):
            return None

        return position

    def compute_ego_trajectory(self):
        """Compute ego trajectory with robust smoothing and interpolation."""
        bbox_df = pd.read_csv(self.bbox_file)

        # Extract traffic light positions with interpolation
        for idx, row in bbox_df.iterrows():
            frame_idx = row['frame']

            if row['x1'] == 0 and row['y1'] == 0:
                # Use last valid position for missing detections
                if self.last_valid_tl_pos is not None:
                    self.traffic_light_positions_cam.append(self.last_valid_tl_pos.copy())
                continue

            xyz_file = self.xyz_path / f"depth{frame_idx:06d}.npz"
            tl_pos = self.extract_3d_position(
                [row['x1'], row['y1'], row['x2'], row['y2']],
                xyz_file, frame_idx
            )

            if tl_pos is not None:
                self.traffic_light_positions_cam.append(tl_pos[:2])
                self.last_valid_tl_pos = tl_pos[:2].copy()
            elif self.last_valid_tl_pos is not None:
                self.traffic_light_positions_cam.append(self.last_valid_tl_pos.copy())

        if not self.traffic_light_positions_cam:
            print("No valid traffic light detections found!")
            return

        # Compute ego trajectory from traffic light observations
        initial_tl_cam = self.traffic_light_positions_cam[0]
        ego_positions = []

        # Initial ego position
        initial_ego_world = np.array([
            -initial_tl_cam[0],
            -initial_tl_cam[1],
        ])
        ego_positions.append(initial_ego_world)

        # Track cumulative motion
        for t in range(1, len(self.traffic_light_positions_cam)):
            delta_tl = self.traffic_light_positions_cam[t] - self.traffic_light_positions_cam[t-1]
            ego_motion = np.array([-delta_tl[0], -delta_tl[1]])
            new_ego_pos = ego_positions[-1] + ego_motion
            ego_positions.append(new_ego_pos)

        self.ego_trajectory = np.array(ego_positions)

        # Apply robust smoothing (same as ego_trajectory.py)
        self.ego_trajectory = self.smooth_trajectory(self.ego_trajectory, method='combined')

        print(f"Computed smooth ego trajectory with {len(self.ego_trajectory)} points")
        if len(self.ego_trajectory) > 0:
            print(f"  Initial position: X={self.ego_trajectory[0,0]:.2f}, Y={self.ego_trajectory[0,1]:.2f}")
            print(f"  Final position: X={self.ego_trajectory[-1,0]:.2f}, Y={self.ego_trajectory[-1,1]:.2f}")

    def detect_objects_in_frame(self, img, xyz_file, frame_idx):
        """Detect objects using YOLO and extract 3D positions."""
        results = self.yolo.track(img, **self.tracker_cfg, verbose=False)

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes

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

                # Apply class-specific confidence threshold
                required_conf = self.confidence_thresholds.get(class_name, 0.1)
                if confidence < required_conf:
                    continue  # Skip detections below threshold

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

    def process_and_smooth_tracks(self):
        """Process all object tracks and apply smoothing."""
        print("\nSmoothing object trajectories...")

        for track_id, track_data in self.object_tracks.items():
            if len(track_data) < 5:
                continue  # Skip very short tracks

            # Get class of this track (majority vote)
            classes = [d['class'] for d in track_data]
            from collections import Counter
            most_common_class = Counter(classes).most_common(1)[0][0]

            # Extract positions and frames
            frames = np.array([d['frame'] for d in track_data])
            positions = np.array([d['position'][:2] for d in track_data])  # Only X,Y

            # Fill gaps in trajectory (interpolation for missing frames)
            if len(frames) > 1:
                # Create full frame range
                full_frames = np.arange(frames.min(), frames.max() + 1)
                full_positions = np.zeros((len(full_frames), 2))

                # Interpolate missing positions
                for dim in range(2):
                    full_positions[:, dim] = np.interp(full_frames, frames, positions[:, dim])

                # Apply smoothing based on object type
                if most_common_class == 'golf_cart':
                    # Golf cart gets aggressive smoothing for clean path
                    smoothed = self.smooth_trajectory(full_positions, method='combined')
                elif most_common_class in ['pedestrian']:
                    # Pedestrians get same aggressive smoothing as ego/cart
                    smoothed = self.smooth_trajectory(full_positions, method='combined')
                elif most_common_class in ['traffic_barrier']:
                    # Static objects get minimal smoothing
                    smoothed = full_positions
                else:
                    # Default smoothing
                    smoothed = self.smooth_trajectory(full_positions, method='savgol')

                self.smoothed_tracks[track_id] = {
                    'class': most_common_class,
                    'frames': full_frames,
                    'positions': smoothed,
                    'original_positions': full_positions
                }

                print(f"  Track {track_id} ({most_common_class}): {len(track_data)} detections -> {len(smoothed)} smoothed points")

    def process_dataset(self):
        """Process entire dataset with smoothing."""
        import cv2
        import time

        # First compute ego trajectory
        print("Computing ego trajectory...")
        self.compute_ego_trajectory()

        if len(self.ego_trajectory) == 0:
            print("Failed to compute ego trajectory")
            return

        # Get all file paths
        rgb_files = sorted(self.rgb_path.glob("*.png"))
        print(f"\nPreloading {len(rgb_files)} frames into memory...")

        # Preload all images and depth data into memory
        start_load = time.time()
        all_images = []
        all_xyz_files = []

        for frame_idx, rgb_file in enumerate(rgb_files):
            if frame_idx % 50 == 0:
                print(f"  Loading frame {frame_idx}/{len(rgb_files)}...")

            # Load image
            img = cv2.imread(str(rgb_file))
            all_images.append(img)

            # Store XYZ file path (we'll load on demand as they're large)
            xyz_file = self.xyz_path / f"depth{frame_idx:06d}.npz"
            all_xyz_files.append(xyz_file)

        load_time = time.time() - start_load
        print(f"Data loaded in {load_time:.1f} seconds")
        print(f"\nProcessing {len(all_images)} frames with YOLO...")

        # Process all frames from memory
        start_process = time.time()
        for frame_idx, (img, xyz_file) in enumerate(zip(all_images, all_xyz_files)):
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_process
                fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"Processing frame {frame_idx}/{len(all_images)} ({fps:.1f} fps)")

            # Detect objects (image already in memory)
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

        # Apply smoothing to all tracks
        self.process_and_smooth_tracks()

        process_time = time.time() - start_process
        total_time = time.time() - start_load
        print(f"\nProcessing complete:")
        print(f"  Loading time: {load_time:.1f}s")
        print(f"  Processing time: {process_time:.1f}s")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {len(all_images)/process_time:.2f}")

        print(f"\nDetection Statistics:")
        class_counts = defaultdict(int)
        for det in self.all_detections:
            class_counts[det['class']] += 1

        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} detections")

        print(f"  Total unique tracks: {len(self.object_tracks)}")
        print(f"  Smoothed tracks: {len(self.smoothed_tracks)}")

    def plot_static_bev(self, save_path: str = "enhanced_bev_smooth.png"):
        """Create static BEV with all smooth trajectories."""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot smooth ego trajectory
        if len(self.ego_trajectory) > 0:
            ax.plot(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1],
                   'b-', linewidth=2.5, label='Ego Trajectory (Smooth)', alpha=0.8)
            ax.scatter(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1],
                      c=np.arange(len(self.ego_trajectory)), cmap='viridis', s=15, alpha=0.5)

            # Mark start and end
            ax.scatter(self.ego_trajectory[0, 0], self.ego_trajectory[0, 1],
                      color='green', s=150, marker='o', label='Start', zorder=5)
            ax.scatter(self.ego_trajectory[-1, 0], self.ego_trajectory[-1, 1],
                      color='red', s=150, marker='s', label='End', zorder=5)

        # Traffic light at origin - determine final state
        # Default to green since that's the most common state
        traffic_light_color = 'lime'
        traffic_light_label = 'Traffic Light (GREEN)'

        # Check traffic light detections to determine state
        tl_detections = [d for d in self.all_detections
                        if d['class'] in ['green_traffic_light', 'red_traffic_light']]
        if tl_detections:
            # Get the most recent traffic light state
            last_tl = max(tl_detections, key=lambda x: x['frame'])
            if last_tl['class'] == 'green_traffic_light':
                traffic_light_color = 'lime'
                traffic_light_label = 'Traffic Light (GREEN)'
            elif last_tl['class'] == 'red_traffic_light':
                traffic_light_color = 'red'
                traffic_light_label = 'Traffic Light (RED)'

        ax.scatter(0, 0, color=traffic_light_color, s=300, marker='*',
                  label=traffic_light_label, zorder=6,
                  edgecolors='black', linewidth=2)

        # Plot traffic light state changes along ego trajectory
        tl_state_changes = []
        for det in self.all_detections:
            if det['class'] in ['green_traffic_light', 'red_traffic_light']:
                frame = det['frame']
                if frame < len(self.ego_trajectory):
                    ego_pos = self.ego_trajectory[frame]
                    tl_state_changes.append({
                        'frame': frame,
                        'position': ego_pos,
                        'color': 'lime' if det['class'] == 'green_traffic_light' else 'red'
                    })

        # Show traffic light states at key points
        if tl_state_changes:
            # Sample every N frames to avoid overcrowding
            sample_rate = max(1, len(tl_state_changes) // 20)
            for i in range(0, len(tl_state_changes), sample_rate):
                state = tl_state_changes[i]
                ax.scatter(state['position'][0], state['position'][1],
                          color=state['color'], marker='o', s=20, alpha=0.6,
                          edgecolors='black', linewidth=0.5)

        # Plot smoothed object trajectories
        for track_id, track_info in self.smoothed_tracks.items():
            class_name = track_info['class']
            smoothed_positions = track_info['positions']

            if class_name in ['green_traffic_light', 'red_traffic_light']:
                continue  # Don't plot traffic light trajectories (they're fixed)

            # Transform to world coordinates
            world_positions = []
            for i, frame in enumerate(track_info['frames']):
                if frame < len(self.ego_trajectory):
                    ego_pos = self.ego_trajectory[frame]
                    world_x = smoothed_positions[i, 0] + ego_pos[0]
                    world_y = smoothed_positions[i, 1] + ego_pos[1]
                    world_positions.append([world_x, world_y])

            if world_positions:
                world_positions = np.array(world_positions)

                # Plot smooth trajectory
                if class_name == 'golf_cart':
                    ax.plot(world_positions[:, 0], world_positions[:, 1],
                           color='gold', linewidth=2, label=f'{class_name} (Smooth)', alpha=0.7)
                    ax.scatter(world_positions[::10, 0], world_positions[::10, 1],
                              color='gold', marker='s', s=30, alpha=0.8)
                elif class_name == 'pedestrian':
                    ax.scatter(world_positions[:, 0], world_positions[:, 1],
                              color='cyan', marker='o', s=20, alpha=0.6, label='Pedestrian')
                elif class_name == 'traffic_barrier':
                    ax.scatter(world_positions[::5, 0], world_positions[::5, 1],
                              color='orange', marker='D', s=40, alpha=0.7, label='Traffic Barriers')

        # Formatting
        ax.set_xlabel('X (meters) - Forward Direction', fontsize=12)
        ax.set_ylabel('Y (meters) - Lateral', fontsize=12)
        ax.set_title('Enhanced BEV with Smooth Trajectories\n(All Objects)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        ax.axis('equal')

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved smooth BEV to {save_path}")
        plt.close()

    def create_animation(self, save_path: str = "enhanced_bev_smooth.mp4"):
        """Create animated BEV with smooth trajectories."""

        fig, ax = plt.subplots(figsize=(10, 10))

        # Set FIXED axis limits for entire animation (no shifting)
        x_min, x_max = -40, 10
        y_min, y_max = -25, 5

        def update_frame(frame_num):
            ax.clear()
            # Track which labels we've already added
            pedestrian_labeled = False
            barrier_labeled = False

            # Reset plot settings
            ax.set_xlabel('X (meters) - Forward Direction', fontsize=12)
            ax.set_ylabel('Y (meters) - Lateral', fontsize=12)
            ax.set_title(f'Enhanced BEV with Smooth Trajectories (Frame {frame_num})\n(Bird\'s Eye View)', fontsize=14)
            ax.grid(True, alpha=0.3)
            # Set aspect ratio and limits without conflicts
            ax.set_xlim(-40, 10)
            ax.set_ylim(-25, 5)
            ax.set_aspect('equal', adjustable='datalim')

            # Reference lines
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            # Plot ego trajectory up to current frame
            if frame_num < len(self.ego_trajectory):
                # Smooth trajectory
                ax.plot(self.ego_trajectory[:frame_num+1, 0],
                       self.ego_trajectory[:frame_num+1, 1],
                       'b-', linewidth=2.5, alpha=0.7, label='Ego Trajectory (Smooth)')

                # Current ego position
                ego_x, ego_y = self.ego_trajectory[frame_num]
                ax.scatter(ego_x, ego_y, color='red', s=200, marker='s',
                          label='Ego Vehicle', zorder=5, edgecolors='darkred', linewidth=2)

            # Mark ego start position
            if len(self.ego_trajectory) > 0:
                ax.scatter(self.ego_trajectory[0, 0], self.ego_trajectory[0, 1],
                          color='green', s=100, marker='o', label='Ego Start', zorder=4,
                          edgecolors='darkgreen', linewidth=2)

            # Traffic light at origin - show current state
            # Default to green (most common state in dataset)
            current_tl_color = 'lime'
            current_tl_label = 'Traffic Light (GREEN)'

            # Find traffic light state for current frame
            frame_tl_detections = [d for d in self.all_detections
                                  if d['frame'] == frame_num and
                                  d['class'] in ['green_traffic_light', 'red_traffic_light']]
            if frame_tl_detections:
                tl_class = frame_tl_detections[0]['class']
                if tl_class == 'green_traffic_light':
                    current_tl_color = 'lime'
                    current_tl_label = 'Traffic Light (GREEN)'
                elif tl_class == 'red_traffic_light':
                    current_tl_color = 'red'
                    current_tl_label = 'Traffic Light (RED)'

            ax.scatter(0, 0, color=current_tl_color, s=250, marker='*',
                      label=current_tl_label, zorder=5,
                      edgecolors='black', linewidth=2)

            # Plot smoothed object positions for current frame
            for track_id, track_info in self.smoothed_tracks.items():
                class_name = track_info['class']

                if class_name in ['green_traffic_light', 'red_traffic_light']:
                    continue

                # Find if this track is active at current frame
                frame_indices = np.where(track_info['frames'] <= frame_num)[0]
                if len(frame_indices) > 0:
                    # Get smoothed trajectory up to current frame
                    track_positions = []
                    for idx in frame_indices:
                        frame = track_info['frames'][idx]
                        if frame < len(self.ego_trajectory):
                            ego_pos = self.ego_trajectory[frame]
                            world_x = track_info['positions'][idx, 0] + ego_pos[0]
                            world_y = track_info['positions'][idx, 1] + ego_pos[1]
                            track_positions.append([world_x, world_y])

                    if track_positions:
                        track_positions = np.array(track_positions)

                        if class_name == 'golf_cart':
                            # Show smooth path
                            ax.plot(track_positions[:, 0], track_positions[:, 1],
                                   color='gold', linewidth=2, alpha=0.6)
                            # Current position
                            ax.scatter(track_positions[-1, 0], track_positions[-1, 1],
                                      color='gold', marker='s', s=150, label='Golf Cart',
                                      edgecolors='darkorange', linewidth=2)
                            # Mark golf cart start position (first detection)
                            ax.scatter(track_positions[0, 0], track_positions[0, 1],
                                      color='gold', marker='D', s=100, label='Cart Start', zorder=4,
                                      edgecolors='darkorange', linewidth=2)
                        elif class_name == 'pedestrian':
                            # Only label once for legend
                            label = 'Pedestrian' if not pedestrian_labeled else None
                            pedestrian_labeled = True
                            ax.scatter(track_positions[-1, 0], track_positions[-1, 1],
                                      color='cyan', marker='o', s=100, alpha=0.8, label=label)
                        elif class_name == 'traffic_barrier':
                            # Show all barrier positions seen so far, only label once
                            label = 'Traffic Barriers' if not barrier_labeled else None
                            barrier_labeled = True
                            ax.scatter(track_positions[::3, 0], track_positions[::3, 1],
                                      color='orange', marker='D', s=50, alpha=0.7, label=label)

            # Create legend with unique entries only
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')

            return ax.patches + ax.lines + ax.collections

        # Create animation
        anim = animation.FuncAnimation(fig, update_frame,
                                     frames=min(len(self.ego_trajectory), 296),
                                     interval=50, blit=False)

        # Save animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Saved smooth animation to {save_path}")
        plt.close()

    def run(self):
        """Main pipeline to process and visualize with smoothing."""
        print("="*60)
        print("Enhanced BEV with Smooth Trajectories")
        print("Applying ego_trajectory.py smoothing to all objects")
        print("="*60)

        # Process dataset
        self.process_dataset()

        # Generate outputs
        self.plot_static_bev("enhanced_bev_smooth.png")
        self.create_animation("enhanced_bev_smooth.mp4")

        print("\n" + "="*60)
        print("Processing complete!")
        print("Outputs:")
        print("  - enhanced_bev_smooth.png (static with all smooth trajectories)")
        print("  - enhanced_bev_smooth.mp4 (animated)")
        print("="*60)


if __name__ == "__main__":
    processor = EnhancedBEVSmooth("dataset")
    processor.run()