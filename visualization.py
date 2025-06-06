import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import argparse
import json
import os


class RobotVisualizer:
    """Simplified robot trajectory and coordinate frame visualizer"""
    
    def __init__(self, extrinsics_path='cam_env/easy_handeye/easy_handeye_eye_on_hand.yaml'):
        """Initialize visualizer with camera extrinsics"""
        self.extrinsics_path = extrinsics_path
        self.ee2camera = self._load_camera_extrinsics()
        
    def _load_camera_extrinsics(self):
        """Load camera extrinsics from YAML file"""
        with open(self.extrinsics_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract transformation parameters
        t = data['transformation']
        quat = [t['qx'], t['qy'], t['qz'], t['qw']]  # [x,y,z,w] format for scipy
        pos = [t['x'], t['y'], t['z']]
        
        # Create transformation matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R.from_quat(quat).as_matrix()
        extrinsics[:3, 3] = pos
        
        return extrinsics
    
    def _load_keypoints(self, rekep_program_dir):
        """Load keypoints from metadata.json"""
        metadata_path = os.path.join(rekep_program_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found at {metadata_path}")
            return []
        
        with open(metadata_path, 'r') as f:
            program_info = json.load(f)
        
        return program_info.get('init_keypoint_positions', [])
    
    def _load_action_sequence(self, action_file_path):
        """Load robot action sequence from JSON file"""
        if not os.path.exists(action_file_path):
            print(f"Warning: Action file not found at {action_file_path}")
            return []
        
        try:
            with open(action_file_path, 'r') as f:
                data = json.load(f)
            
            if 'ee_action_seq' not in data:
                print(f"Warning: Invalid action file format")
                return []
            
            actions = data['ee_action_seq']
            print(f"Loaded {len(actions)} actions from {action_file_path}")
            return actions
        except Exception as e:
            print(f"Error loading action sequence: {e}")
            return []
    
    def _create_coordinate_frame(self, transform, scale=0.05):
        """Create coordinate frame vectors from transformation matrix"""
        origin = transform[:3, 3]
        
        # X, Y, Z axes vectors
        x_axis = origin + scale * transform[:3, 0]
        y_axis = origin + scale * transform[:3, 1] 
        z_axis = origin + scale * transform[:3, 2]
        
        return origin, x_axis, y_axis, z_axis
    
    def _draw_coordinate_frame(self, ax, transform, scale=0.05, label=None):
        """Draw coordinate frame on 3D plot"""
        origin, x_axis, y_axis, z_axis = self._create_coordinate_frame(transform, scale)
        
        # Draw axes with standard colors
        ax.quiver(*origin, *(x_axis - origin), color='r', arrow_length_ratio=0.1)
        ax.quiver(*origin, *(y_axis - origin), color='g', arrow_length_ratio=0.1)  
        ax.quiver(*origin, *(z_axis - origin), color='b', arrow_length_ratio=0.1)
        
        if label:
            ax.text(*origin, label, fontsize=12)
    
    def _transform_keypoints_to_world(self, keypoints, ee_pose=None):
        """Transform keypoints from camera to world coordinates (eye-to-hand setup)"""
        keypoints = np.array(keypoints)
        
        # Load camera extrinsics (now base2camera)
        base2camera = self.ee2camera  # Reuse the loaded extrinsics, now represents base2camera
        
        # Convert to homogeneous coordinates
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        
        # Apply transformation directly (camera to world/base)
        world_coords = (base2camera @ keypoints_homogeneous.T).T
        
        # Convert back to 3D coordinates
        return world_coords[:, :3] / world_coords[:, 3, np.newaxis]
    
    def _get_test_ee_pose(self):
        """Get test end effector pose for visualization"""
        return np.array([-0.25736064, -0.16367309, 0.56684215, 0.99999998, -0.00002284, 0.00000602, 2.11842992e-04])
    
    def visualize(self, rekep_program_dir=None, action_file_path=None):
        """Main visualization function"""
        # Setup the plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get end effector pose
        ee_pose = self._get_test_ee_pose()
        
        # Create transformation matrices
        base_frame = np.eye(4)
        ee_frame = self._create_ee_frame(ee_pose)
        camera_frame = self._create_camera_frame(ee_frame)
        
        # Draw coordinate frames
        self._draw_coordinate_frame(ax, base_frame, scale=0.1, label='Base')
        self._draw_coordinate_frame(ax, ee_frame, scale=0.08, label='End Effector')
        self._draw_coordinate_frame(ax, camera_frame, scale=0.05, label='Camera')
        
        # Collect points for setting plot limits
        all_points = [base_frame[:3, 3], ee_frame[:3, 3], camera_frame[:3, 3]]
        
        # Visualize action trajectory if provided
        if action_file_path:
            trajectory_points = self._visualize_trajectory(ax, action_file_path)
            if trajectory_points is not None:
                all_points.extend(trajectory_points)
        
        # Visualize keypoints if provided
        if rekep_program_dir:
            keypoint_points = self._visualize_keypoints(ax, rekep_program_dir, ee_pose, camera_frame)
            if keypoint_points is not None:
                all_points.extend(keypoint_points)
        
        # Setup plot appearance
        self._setup_plot(ax, all_points)
        
        # Save and show
        plt.savefig('robot_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _create_ee_frame(self, ee_pose):
        """Create end effector transformation matrix"""
        position = ee_pose[:3]
        quat = np.array([ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])  # [w,x,y,z] -> [x,y,z,w]
        rotation = R.from_quat(quat).as_matrix()
        
        ee_frame = np.eye(4)
        ee_frame[:3, :3] = rotation
        ee_frame[:3, 3] = position
        
        return ee_frame
    
    def _create_camera_frame(self, ee_frame):
        """Create camera transformation matrix"""
        return ee_frame @ self.ee2camera
    
    def _visualize_trajectory(self, ax, action_file_path):
        """Visualize robot trajectory"""
        actions = self._load_action_sequence(action_file_path)
        if not actions:
            return None
        
        # Extract positions and gripper states
        positions = np.array([action[:3] for action in actions])
        gripper_states = np.array([action[7] if len(action) > 7 else 0 for action in actions])
        
        # Plot trajectory line
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               'g-', linewidth=2, label='Robot Trajectory')
        
        # Plot waypoints with gripper state colors
        for i, (pos, gripper) in enumerate(zip(positions[::5], gripper_states[::5])):  # Show every 5th point
            color = 'red' if gripper > 0.5 else 'green'
            marker = 'o' if gripper > 0.5 else '^'
            ax.scatter(pos[0], pos[1], pos[2], color=color, s=50, marker=marker)
        
        # Add legend
        ax.scatter([], [], color='red', s=50, marker='o', label='Gripper Closed')
        ax.scatter([], [], color='green', s=50, marker='^', label='Gripper Open')
        
        print(f"Visualized trajectory with {len(positions)} waypoints")
        return positions.tolist()
    
    def _visualize_keypoints(self, ax, rekep_program_dir, ee_pose, camera_frame):
        """Visualize keypoints in both camera and world frames"""
        keypoints_camera = self._load_keypoints(rekep_program_dir)
        if not keypoints_camera:
            return None
        
        print(f"Loaded {len(keypoints_camera)} keypoints")
        
        # Transform keypoints to world coordinates
        keypoints_world = self._transform_keypoints_to_world(keypoints_camera, ee_pose)
        
        # Visualize keypoints in camera frame (using base2camera directly)
        keypoints_homogeneous = np.hstack((keypoints_camera, np.ones((len(keypoints_camera), 1))))
        keypoints_in_world = (self.ee2camera @ keypoints_homogeneous.T).T
        keypoints_in_world = keypoints_in_world[:, :3] / keypoints_in_world[:, 3, np.newaxis]
        
        # Plot keypoints
        ax.scatter(keypoints_in_world[:, 0], keypoints_in_world[:, 1], keypoints_in_world[:, 2], 
                  color='purple', s=100, label='Keypoints (Camera Frame)')
        ax.scatter(keypoints_world[:, 0], keypoints_world[:, 1], keypoints_world[:, 2], 
                  color='orange', s=100, label='Keypoints (World Frame)')
        
        # Add keypoint labels
        for i, (kp_cam, kp_world) in enumerate(zip(keypoints_in_world, keypoints_world)):
            ax.text(kp_cam[0], kp_cam[1], kp_cam[2], f"{i}", color='purple', fontsize=10)
            ax.text(kp_world[0], kp_world[1], kp_world[2], f"{i}", color='orange', fontsize=10)
        
        # Return all keypoint positions for plot limits
        all_keypoints = np.vstack([keypoints_in_world, keypoints_world])
        return all_keypoints.tolist()
    
    def _setup_plot(self, ax, all_points):
        """Setup plot appearance and limits"""
        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Calculate plot limits
        all_points = np.array(all_points)
        margin = 0.1
        min_vals = np.min(all_points, axis=0) - margin
        max_vals = np.max(all_points, axis=0) + margin
        
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])
        
        # Set view angle and add legend
        ax.view_init(elev=20, azim=45)
        ax.legend()
        ax.set_title('Robot Visualization: Frames, Keypoints, and Trajectory')


def find_latest_directory(base_dir):
    """Find the most recently modified directory"""
    if not os.path.exists(base_dir):
        return None
    
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]
    
    return max(dirs, key=os.path.getmtime) if dirs else None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize robot frames, keypoints, and trajectory')
    parser.add_argument('--extrinsics', type=str, 
                       default='cam_env/easy_handeye/easy_handeye_eye_on_hand.yaml',
                       help='Path to camera extrinsics YAML file')
    parser.add_argument('--rekep_dir', type=str, 
                       help='Path to ReKep program directory containing metadata.json')
    parser.add_argument('--action_file', type=str, 
                       help='Path to action.json file containing robot trajectory')
    args = parser.parse_args()
    
    # Auto-find directories if not provided
    if not args.rekep_dir:
        args.rekep_dir = find_latest_directory("./vlm_query/")
        if args.rekep_dir:
            print(f"\033[92mUsing latest ReKep directory: {args.rekep_dir}\033[0m")
    
    if not args.action_file:
        default_action_file = "./outputs/action.json"
        if os.path.exists(default_action_file):
            args.action_file = default_action_file
            print(f"\033[92mUsing action file: {args.action_file}\033[0m")
    
    # Create visualizer and run
    visualizer = RobotVisualizer(args.extrinsics)
    visualizer.visualize(args.rekep_dir, args.action_file)


if __name__ == "__main__":
    main()