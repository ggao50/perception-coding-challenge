import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import cv2
from typing import Tuple, List
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')


class EgoTrajectoryEstimator:
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.rgb_path = self.dataset_path / "rgb"
        self.xyz_path = self.dataset_path / "xyz"
        self.bbox_file = self.dataset_path / "bboxes_light.csv"

        self.traffic_light_positions_cam = []
        self.ego_trajectory = []
        self.world_origin = None
        self.initial_rotation = None

    def load_traffic_light_bboxes(self) -> pd.DataFrame:
        """Load traffic light bounding boxes from CSV."""
        df = pd.read_csv(self.bbox_file)
        print(f"Loaded {len(df)} frames of bounding box data")
        return df

    def get_traffic_light_3d_position(self, frame_idx: int, bbox_row: pd.Series) -> np.ndarray:
        """Extract 3D position of traffic light from depth data."""
        x1, y1, x2, y2 = bbox_row['x1'], bbox_row['y1'], bbox_row['x2'], bbox_row['y2']

        # Skip frames with no detection (all zeros)
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            return None

        # Get center of bounding box
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # Load depth data
        depth_file = self.xyz_path / f"depth{frame_idx:06d}.npz"
        if not depth_file.exists():
            return None

        xyz_data = np.load(depth_file)["xyz"]  # Shape (H, W, 4)
        xyz = xyz_data[:, :, :3]  # Take only X, Y, Z (first 3 channels)

        # Get 3D position at pixel location
        # Use a small patch around center for robustness
        patch_size = 5
        v_min = max(0, v - patch_size)
        v_max = min(xyz.shape[0], v + patch_size)
        u_min = max(0, u - patch_size)
        u_max = min(xyz.shape[1], u + patch_size)

        patch = xyz[v_min:v_max, u_min:u_max]

        # Filter out invalid points (0 or NaN or Inf)
        valid_mask = (patch[:,:,0] != 0) & np.isfinite(patch[:,:,0]) & np.isfinite(patch[:,:,1]) & np.isfinite(patch[:,:,2])
        if not np.any(valid_mask):
            # Fallback to center point if valid
            center_point = xyz[v, u]
            if np.all(np.isfinite(center_point)) and center_point[0] != 0:
                return center_point
            else:
                return None

        # Average valid points in patch
        valid_points = patch[valid_mask]
        position = np.mean(valid_points, axis=0)

        # Final check for validity
        if not np.all(np.isfinite(position)):
            return None

        return position

    def smooth_trajectory(self, trajectory: np.ndarray, method: str = 'combined') -> np.ndarray:
        """Smooth the trajectory using various filtering methods."""
        if len(trajectory) < 5:
            return trajectory

        smoothed = trajectory.copy()

        if method == 'savgol':
            # Savitzky-Golay filter for each dimension
            window_length = min(31, len(trajectory) if len(trajectory) % 2 == 1 else len(trajectory) - 1)
            window_length = max(5, window_length)  # Ensure minimum window size

            smoothed[:, 0] = savgol_filter(trajectory[:, 0], window_length, 3)
            smoothed[:, 1] = savgol_filter(trajectory[:, 1], window_length, 3)

        elif method == 'moving_average':
            # Simple moving average
            window = 15
            for i in range(len(trajectory)):
                start = max(0, i - window // 2)
                end = min(len(trajectory), i + window // 2 + 1)
                smoothed[i] = np.mean(trajectory[start:end], axis=0)

        elif method == 'spline':
            # Fit a spline to the trajectory
            t = np.arange(len(trajectory))
            # Use lower smoothing factor for more aggressive smoothing
            spl_x = UnivariateSpline(t, trajectory[:, 0], s=50)
            spl_y = UnivariateSpline(t, trajectory[:, 1], s=50)
            smoothed[:, 0] = spl_x(t)
            smoothed[:, 1] = spl_y(t)

        elif method == 'combined':
            # First apply moving average to remove high-frequency noise
            window = 11
            temp = trajectory.copy()
            for i in range(len(trajectory)):
                start = max(0, i - window // 2)
                end = min(len(trajectory), i + window // 2 + 1)
                temp[i] = np.mean(trajectory[start:end], axis=0)

            # Then apply Savitzky-Golay for smooth derivatives
            window_length = min(51, len(temp) if len(temp) % 2 == 1 else len(temp) - 1)
            window_length = max(5, window_length)

            smoothed[:, 0] = savgol_filter(temp[:, 0], window_length, 3)
            smoothed[:, 1] = savgol_filter(temp[:, 1], window_length, 3)

            # Finally, ensure the trajectory is monotonic in X (always moving forward)
            # This creates a straighter path
            for i in range(1, len(smoothed)):
                if smoothed[i, 0] < smoothed[i-1, 0]:
                    # If we're going backward, interpolate
                    smoothed[i, 0] = smoothed[i-1, 0] + 0.01

        elif method == 'linear_fit':
            # Fit a straight line (most aggressive smoothing)
            t = np.arange(len(trajectory))
            # Fit linear polynomial for X and Y
            coef_x = np.polyfit(t, trajectory[:, 0], 1)
            coef_y = np.polyfit(t, trajectory[:, 1], 1)
            smoothed[:, 0] = np.polyval(coef_x, t)
            smoothed[:, 1] = np.polyval(coef_y, t)

        return smoothed

    def transform_to_world_frame(self, point_cam: np.ndarray, t: int) -> np.ndarray:
        """Transform a point from camera frame to world frame."""
        if t == 0:
            # At t=0, define world frame
            # Origin is under traffic light on ground
            # X-axis points from origin towards car
            # Z-axis points up through traffic light

            # Traffic light position in camera frame at t=0
            tl_cam = self.traffic_light_positions_cam[0]

            # World origin is at ground level under traffic light
            self.world_origin = np.array([tl_cam[0], tl_cam[1], 0])

            # Initial rotation aligns camera X with world -X (car looking at light)
            # This makes the line from car to light align with +X in world
            self.initial_rotation = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])

        # Transform: world_point = R * (cam_point - origin)
        world_point = self.initial_rotation @ (point_cam - self.world_origin)
        return world_point

    def compute_ego_trajectory(self, bbox_df: pd.DataFrame):
        """Compute ego vehicle trajectory from traffic light observations."""
        print("Computing ego trajectory...")

        # First pass: collect all valid traffic light positions in camera frame
        last_valid_pos = None
        for idx, row in bbox_df.iterrows():
            frame_idx = row['frame']
            tl_pos_cam = self.get_traffic_light_3d_position(frame_idx, row)

            if tl_pos_cam is not None:
                self.traffic_light_positions_cam.append(tl_pos_cam)
                last_valid_pos = tl_pos_cam
            else:
                # Use last known valid position if available
                if last_valid_pos is not None:
                    self.traffic_light_positions_cam.append(last_valid_pos.copy())
                else:
                    # Skip frames until we get first valid detection
                    continue

        # Define world frame based on first observation
        if not self.traffic_light_positions_cam:
            print("No valid traffic light detections found!")
            return

        # The traffic light is fixed in world at (0, 0, height).
        # The ego vehicle moves toward it.
        #
        # In camera coordinates:
        # +X = forward (from camera)
        # +Y = right
        # +Z = up
        #
        # World frame: origin at ground under traffic light
        # The ego starts behind and approaches the light

        # Get initial traffic light position to establish reference
        initial_tl_cam = self.traffic_light_positions_cam[0]

        # Compute ego trajectory using cumulative motion
        ego_positions = []

        # Initial ego position: behind the traffic light
        # The traffic light is at (0,0) in world, ego starts at negative X
        # World frame: +Y is LEFT (opposite of camera's +Y which is RIGHT)
        initial_ego_world = np.array([
            -initial_tl_cam[0],  # Ego starts behind (negative X)
            -initial_tl_cam[1],  # Camera +Y (right) -> World -Y (ego is right of TL = negative Y in world where +Y is left)
        ])
        ego_positions.append(initial_ego_world)

        # Track cumulative motion
        for t in range(1, len(self.traffic_light_positions_cam)):
            # Change in traffic light position in camera frame
            delta_tl = self.traffic_light_positions_cam[t] - self.traffic_light_positions_cam[t-1]

            # Ego motion is opposite to apparent traffic light motion
            ego_motion = np.array([
                -delta_tl[0],  # Forward motion (opposite of TL motion)
                -delta_tl[1],  # Lateral motion (opposite due to both inversion and axis flip)
            ])

            # Update ego position
            new_ego_pos = ego_positions[-1] + ego_motion
            ego_positions.append(new_ego_pos)

        self.ego_trajectory = np.array(ego_positions)

        # Smooth the trajectory
        self.ego_trajectory = self.smooth_trajectory(self.ego_trajectory, method='combined')

        print(f"Computed trajectory with {len(self.ego_trajectory)} points")

        # Print some debug info
        if len(self.ego_trajectory) > 0:
            print(f"  Initial position: X={self.ego_trajectory[0,0]:.2f}, Y={self.ego_trajectory[0,1]:.2f}")
            print(f"  Final position: X={self.ego_trajectory[-1,0]:.2f}, Y={self.ego_trajectory[-1,1]:.2f}")
            initial_dist = np.linalg.norm(self.ego_trajectory[0])
            final_dist = np.linalg.norm(self.ego_trajectory[-1])
            print(f"  Initial distance to light: {initial_dist:.2f}m")
            print(f"  Final distance to light: {final_dist:.2f}m")

    def plot_trajectory(self, save_path: str = "trajectory.png"):
        """Plot the ego vehicle trajectory in bird's eye view."""
        if len(self.ego_trajectory) == 0:
            print("No trajectory to plot!")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot trajectory
        ax.plot(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1],
                'b-', linewidth=2, label='Ego Trajectory')
        ax.scatter(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1],
                  c=np.arange(len(self.ego_trajectory)), cmap='viridis', s=20)

        # Mark start and end
        ax.scatter(self.ego_trajectory[0, 0], self.ego_trajectory[0, 1],
                  color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(self.ego_trajectory[-1, 0], self.ego_trajectory[-1, 1],
                  color='red', s=100, marker='s', label='End', zorder=5)

        # Mark traffic light position (origin)
        ax.scatter(0, 0, color='yellow', s=200, marker='*',
                  label='Traffic Light (World Origin)', zorder=5, edgecolors='black', linewidth=2)

        # Formatting
        ax.set_xlabel('X (meters) - Forward Direction', fontsize=12)
        ax.set_ylabel('Y (meters) - Lateral', fontsize=12)
        ax.set_title('Ego Vehicle Trajectory in Ground Frame\n(Bird\'s Eye View)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

        # Add annotations to show car approaching traffic light
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.text(0.5, -1, 'Traffic Light\nLocation', fontsize=10, ha='left', color='gray')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
        plt.close()

    def create_animation(self, save_path: str = "trajectory.mp4"):
        """Create animated trajectory video."""
        if len(self.ego_trajectory) == 0:
            print("No trajectory to animate!")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Set up plot limits with safety checks
        x_vals = self.ego_trajectory[:, 0]
        y_vals = self.ego_trajectory[:, 1]

        # Remove any NaN or Inf values
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(valid_mask):
            print("No valid trajectory points for animation!")
            return

        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]

        x_min, x_max = x_vals.min() - 5, x_vals.max() + 5
        y_min, y_max = y_vals.min() - 5, y_vals.max() + 5

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (meters) - Forward', fontsize=12)
        ax.set_ylabel('Y (meters) - Left', fontsize=12)
        ax.set_title('Ego Vehicle Trajectory Animation\n(Bird\'s Eye View)', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Traffic light at origin
        ax.scatter(0, 0, color='yellow', s=200, marker='*',
                  label='Traffic Light', zorder=5, edgecolors='black', linewidth=2)

        # Initialize trajectory line
        line, = ax.plot([], [], 'b-', linewidth=2, label='Trajectory')
        point, = ax.plot([], [], 'ro', markersize=10)

        # Trail points
        trail_scatter = ax.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.6)

        ax.legend()

        def init():
            line.set_data([], [])
            point.set_data([], [])
            trail_scatter.set_offsets(np.empty((0, 2)))
            return line, point, trail_scatter

        def animate(frame):
            # Update trajectory line
            line.set_data(self.ego_trajectory[:frame+1, 0],
                         self.ego_trajectory[:frame+1, 1])

            # Update current position
            point.set_data([self.ego_trajectory[frame, 0]],
                          [self.ego_trajectory[frame, 1]])

            # Update trail points with color gradient
            if frame > 0:
                trail_scatter.set_offsets(self.ego_trajectory[:frame+1])
                trail_scatter.set_array(np.arange(frame+1))

            # Update title with frame number
            ax.set_title(f'Ego Vehicle Trajectory Animation\n(Frame {frame}/{len(self.ego_trajectory)-1})',
                        fontsize=14)

            return line, point, trail_scatter

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(self.ego_trajectory),
                                     interval=50, blit=False)

        # Save animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Saved trajectory animation to {save_path}")
        plt.close()

    def run(self):
        """Main pipeline to compute and visualize ego trajectory."""
        print("Starting ego trajectory estimation...")

        # Load bounding boxes
        bbox_df = self.load_traffic_light_bboxes()

        # Compute trajectory
        self.compute_ego_trajectory(bbox_df)

        # Generate outputs
        self.plot_trajectory("trajectory.png")
        self.create_animation("trajectory.mp4")

        print("Ego trajectory estimation complete!")

        # Print some statistics
        if len(self.ego_trajectory) > 0:
            total_distance = np.sum(np.linalg.norm(np.diff(self.ego_trajectory, axis=0), axis=1))
            print(f"\nTrajectory Statistics:")
            print(f"  Total frames: {len(self.ego_trajectory)}")
            print(f"  Total distance traveled: {total_distance:.2f} meters")
            print(f"  Average speed: {total_distance / (len(self.ego_trajectory) / 30):.2f} m/s (assuming 30 fps)")


if __name__ == "__main__":
    estimator = EgoTrajectoryEstimator("dataset")
    estimator.run()