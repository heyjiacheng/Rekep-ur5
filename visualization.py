import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import argparse
import json
import os

def load_camera_extrinsics(extrinsics_path):
    """Load camera extrinsics from YAML file"""
    with open(extrinsics_path, 'r') as f:
        extrinsics_data = yaml.safe_load(f)
    
    # Extract transformation parameters
    qw = extrinsics_data['transformation']['qw']
    qx = extrinsics_data['transformation']['qx']
    qy = extrinsics_data['transformation']['qy']
    qz = extrinsics_data['transformation']['qz']
    tx = extrinsics_data['transformation']['x']
    ty = extrinsics_data['transformation']['y']
    tz = extrinsics_data['transformation']['z']
    
    # Create rotation matrix from quaternion
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # Create 4x4 transformation matrix
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot
    extrinsics[:3, 3] = [tx, ty, tz]
    
    return extrinsics

def draw_coordinate_frame(ax, transform, scale=0.05, label=None):
    """Draw a coordinate frame based on a transformation matrix"""
    origin = transform[:3, 3]
    
    # X, Y, Z axes
    x_axis = origin + scale * transform[:3, 0]
    y_axis = origin + scale * transform[:3, 1]
    z_axis = origin + scale * transform[:3, 2]
    
    # Draw axes
    ax.quiver(*origin, *(x_axis - origin), color='r', label='X' if label else None)
    ax.quiver(*origin, *(y_axis - origin), color='g', label='Y' if label else None)
    ax.quiver(*origin, *(z_axis - origin), color='b', label='Z' if label else None)
    
    if label:
        ax.text(*origin, label, fontsize=12)

def transform_keypoints_to_world(keypoints, ee_pose, ee2camera):
    """
    Transform keypoints from camera coordinate system to world coordinate system
    """
    # Convert to numpy array
    keypoints = np.array(keypoints)
    
    # Convert to homogeneous coordinates
    keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
    
    # EE frame with handedness correction
    position = ee_pose[:3]
    quat = np.array([ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])  # [qx,qy,qz,qw]
    rotation = R.from_quat(quat).as_matrix()
    
    # Apply handedness correction - reverse X and Z axes
    rot_correct = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    rotation_corrected = rotation @ rot_correct
    
    # Original EE transformation matrix (without offset)
    base2ee_original = np.eye(4)
    base2ee_original[:3, :3] = rotation_corrected
    base2ee_original[:3, 3] = position
    
    # Camera frame based on original EE position
    camera_frame_incorrect = base2ee_original @ ee2camera
    
    # Create camera axes correction matrix
    camera_axes_correction = np.array([
        [0, 0, 1],  # New x-axis is old z-axis
        [-1, 0, 0], # New y-axis is old x-axis
        [0, -1, 0]  # New z-axis is negative old y-axis
    ])
    
    # Apply the correction to the camera frame rotation part
    camera_frame = camera_frame_incorrect.copy()
    camera_frame[:3, :3] = camera_frame_incorrect[:3, :3] @ camera_axes_correction
    
    # Apply transformation
    base_coords_homogeneous = (camera_frame @ keypoints_homogeneous.T).T
    
    # Convert back to non-homogeneous coordinates
    base_coords = base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]
    
    return base_coords

def load_keypoints(rekep_program_dir):
    """Load keypoints from metadata.json"""
    metadata_path = os.path.join(rekep_program_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}")
        return []
    
    with open(metadata_path, 'r') as f:
        program_info = json.load(f)
    
    return program_info.get('init_keypoint_positions', [])

def load_action_sequence(action_file_path):
    """Load action sequence from JSON file"""
    if not os.path.exists(action_file_path):
        print(f"Warning: Action file not found at {action_file_path}")
        return []
    
    try:
        with open(action_file_path, 'r') as f:
            data = json.load(f)
        
        if 'ee_action_seq' not in data:
            print(f"Warning: Invalid action file format: missing 'ee_action_seq'")
            return []
        
        print(f"Loaded {len(data['ee_action_seq'])} actions from {action_file_path}")
        return data['ee_action_seq']
    except Exception as e:
        print(f"Error loading action sequence: {e}")
        return []

def visualize_frames(rekep_program_dir=None, action_file_path=None):
    """Visualize the base, EE, and camera coordinate frames with the correct interpretation"""
    # Load camera extrinsics
    extrinsics_path = '/home/xu/.ros/easy_handeye/easy_handeye_eye_on_hand.yaml'
    ee2camera = load_camera_extrinsics(extrinsics_path)
    
    # Get test EE pose
    ee_pose = np.array([-0.3029407, -0.10364389, 0.47234771, -0.99996402, 0.00576533, 0.00473004, -0.00404257])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Base frame is identity
    base_frame = np.eye(4)
    
    # EE frame with handedness correction
    position = ee_pose[:3]
    quat = np.array([ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])  # [qx,qy,qz,qw]
    rotation = R.from_quat(quat).as_matrix()
    
    # Apply handedness correction - reverse X and Z axes
    rot_correct = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    rotation_corrected = rotation @ rot_correct
    
    # Original EE transformation matrix (without offset)
    base2ee_original = np.eye(4)
    base2ee_original[:3, :3] = rotation_corrected
    base2ee_original[:3, 3] = position
    
    # Calculate offset along the corrected EE z-axis for gripper
    z_offset = np.array([0, 0, 0.14])  # 0.14m along z-axis
    z_offset_world = rotation_corrected @ z_offset
    
    # EE frame with gripper offset
    base2ee_with_gripper = np.eye(4)
    base2ee_with_gripper[:3, :3] = rotation_corrected
    base2ee_with_gripper[:3, 3] = position + z_offset_world
    
    # Camera frame based on original EE position
    camera_frame_incorrect = base2ee_original @ ee2camera
    
    # Create camera axes correction matrix
    camera_axes_correction = np.array([
        [0, 0, 1],  # New x-axis is old z-axis
        [-1, 0, 0], # New y-axis is old x-axis
        [0, -1, 0]  # New z-axis is negative old y-axis
    ])
    
    # Apply the correction to the camera frame rotation part
    camera_frame = camera_frame_incorrect.copy()
    camera_frame[:3, :3] = camera_frame_incorrect[:3, :3] @ camera_axes_correction
    
    # Draw coordinate frames
    draw_coordinate_frame(ax, base_frame, scale=0.1, label='Base')
    draw_coordinate_frame(ax, base2ee_original, scale=0.08, label='EE')
    draw_coordinate_frame(ax, base2ee_with_gripper, scale=0.08, label='EE with Gripper')
    draw_coordinate_frame(ax, camera_frame, scale=0.05, label='Camera')
    
    # Load and visualize action sequence if provided
    if action_file_path:
        action_sequence = load_action_sequence(action_file_path)
        if action_sequence:
            # Extract positions and gripper states
            positions = np.array([action[:3] for action in action_sequence])
            gripper_states = np.array([action[6] for action in action_sequence])
            
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'g-', linewidth=2, label='Robot Trajectory')
            
            # Plot waypoints
            for i, (pos, gripper) in enumerate(zip(positions, gripper_states)):
                # Color based on gripper state (red for closed, green for open)
                color = 'r' if gripper > 0.5 else 'g'
                marker = 'o' if gripper > 0.5 else '^'
                ax.scatter(pos[0], pos[1], pos[2], color=color, s=100, marker=marker)
                ax.text(pos[0], pos[1], pos[2], f"{i}", color='black', fontsize=10)
            
            # Add legend for gripper states
            ax.scatter([], [], color='r', s=100, marker='o', label='Gripper Closed')
            ax.scatter([], [], color='g', s=100, marker='^', label='Gripper Open')
            
            print(f"Visualized {len(action_sequence)} action waypoints")
    
    # Load and transform keypoints if rekep_program_dir is provided
    if rekep_program_dir:
        keypoints_camera = load_keypoints(rekep_program_dir)
        if keypoints_camera:
            print(f"Loaded {len(keypoints_camera)} keypoints from {rekep_program_dir}")
            print(f"Camera keypoints: {keypoints_camera}")
            
            # Transform keypoints to world coordinates
            keypoints_world = transform_keypoints_to_world(keypoints_camera, ee_pose, ee2camera)
            print(f"World keypoints: {keypoints_world}")
            
            # Plot keypoints in camera frame (as small spheres)
            keypoints_camera_homogeneous = np.hstack((keypoints_camera, np.ones((len(keypoints_camera), 1))))
            keypoints_camera_in_world = (camera_frame @ keypoints_camera_homogeneous.T).T
            keypoints_camera_in_world = keypoints_camera_in_world[:, :3] / keypoints_camera_in_world[:, 3, np.newaxis]
            
            ax.scatter(keypoints_camera_in_world[:, 0], keypoints_camera_in_world[:, 1], 
                      keypoints_camera_in_world[:, 2], color='purple', s=100, label='Keypoints (Camera Frame)')
            
            # Plot transformed keypoints in world frame
            ax.scatter(keypoints_world[:, 0], keypoints_world[:, 1], keypoints_world[:, 2], 
                      color='orange', s=100, label='Keypoints (World Frame)')
            
            # Add keypoint indices as text
            for i, (kp_cam, kp_world) in enumerate(zip(keypoints_camera_in_world, keypoints_world)):
                ax.text(kp_cam[0], kp_cam[1], kp_cam[2], f"{i}", color='purple', fontsize=12)
                ax.text(kp_world[0], kp_world[1], kp_world[2], f"{i}", color='orange', fontsize=12)
    
    # Collect all points for setting plot limits
    all_points = np.vstack([
        base_frame[:3, 3],
        base2ee_original[:3, 3],
        base2ee_with_gripper[:3, 3],
        camera_frame[:3, 3]
    ])
    
    # Include keypoints in limits calculation if available
    if rekep_program_dir and 'keypoints_world' in locals():
        all_points = np.vstack([all_points, keypoints_world])
    
    # Include action trajectory in limits calculation if available
    if action_file_path and 'positions' in locals() and len(positions) > 0:
        all_points = np.vstack([all_points, positions])
    
    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set limits appropriately
    min_vals = np.min(all_points, axis=0) - 0.1
    max_vals = np.max(all_points, axis=0) + 0.1
    
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])
    
    ax.legend()
    
    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.title('Visualization of Base, EE, Camera Frames and Action Trajectory')
    plt.tight_layout()
    plt.savefig('visualization.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize coordinate frames and keypoints')
    parser.add_argument('--extrinsics', type=str, default='/home/xu/.ros/easy_handeye/easy_handeye_eye_on_hand.yaml',
                      help='Path to camera extrinsics YAML file')
    parser.add_argument('--rekep_dir', type=str, help='Path to ReKep program directory containing metadata.json')
    parser.add_argument('--action_file', type=str, help='Path to action.json file containing robot trajectory')
    args = parser.parse_args()
    
    # If rekep_dir is not provided, try to find the most recent directory
    if not args.rekep_dir:
        vlm_query_dir = "./vlm_query/"
        if os.path.exists(vlm_query_dir):
            vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                        if os.path.isdir(os.path.join(vlm_query_dir, d))]
            if vlm_dirs:
                args.rekep_dir = max(vlm_dirs, key=os.path.getmtime)
                print(f"\033[92mUsing most recent directory: {args.rekep_dir}\033[0m")
    
    # If action_file is not provided, try to use the default location
    if not args.action_file:
        default_action_file = "./outputs/action.json"
        if os.path.exists(default_action_file):
            args.action_file = default_action_file
            print(f"\033[92mUsing action file: {args.action_file}\033[0m")
    
    visualize_frames(args.rekep_dir, args.action_file)

if __name__ == "__main__":
    main()