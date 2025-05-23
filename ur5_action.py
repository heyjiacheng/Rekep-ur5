import torch
import numpy as np
import json
import os
import sys
from scipy.spatial.transform import Rotation as R
import yaml
from ur_env.rotations import pose2quat, quat_2_rotvec
import argparse

from rekep.environment import R2D2Env
from rekep.ik_solver import UR5IKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
from rekep.visualizer import Visualizer
from ur_env.ur5_env import RobotEnv
from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")


class UR5Controller:
    """Simplified UR5 robot controller for executing manipulation tasks"""
    
    def __init__(self, visualize=False):
        """Initialize UR5 controller"""
        self.config = self._load_config()
        self.visualize = visualize
        self._setup_components()
        self._set_random_seeds()
        
        # Store initial position for returning after task completion
        self.initial_position = None
        
    def _load_config(self):
        """Load configuration file"""
        global_config = get_config(config_path="./configs/config.yaml")
        return global_config['main']
    
    def _setup_components(self):
        """Setup robot components and solvers"""
        # Robot environments
        self.robot_env = RobotEnv()
        global_config = get_config(config_path="./configs/config.yaml")
        self.env = R2D2Env(global_config['env'])
        
        # Default reset joint positions for UR5
        self.reset_joint_pos = np.array([
            -0.023413960133687794, -1.9976251761065882, 1.7851085662841797, 
            4.942904949188232, -1.5486105124102991, -1.5801880995379847
        ])
        
        # IK solver
        ik_solver = UR5IKSolver(
            reset_joint_pos=self.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        
        # Motion solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.reset_joint_pos)
        
        # Visualizer
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'])
            self.data_path = "/home/franka/R2D2_3dhat/images/current_images"
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def execute_task(self, instruction=None, rekep_program_dir=None):
        """Execute robot manipulation task"""
        
        if rekep_program_dir is None:
            print("Error: Need to provide rekep program directory")
            return
            
        self._run_task(rekep_program_dir)

    def _run_task(self, rekep_program_dir):
        """Main task execution loop"""
        # Store initial robot position
        self.initial_position = self.robot_env.robot.get_tcp_pose()
        
        # Load program configuration
        self._load_program_info(rekep_program_dir)
        self._setup_task_environment()
        
        # Start from stage 1
        self._update_stage(1)
        
        # Get scene keypoints and transform to world coordinates
        scene_keypoints = self._get_world_keypoints()
        
        # Main execution loop
        while True:
            # Update current state
            self._update_current_state(scene_keypoints)
            
            # Generate next subgoal and path
            next_subgoal = self._generate_subgoal()
            next_path = self._generate_path(next_subgoal)
            
            # Execute the planned path
            self._execute_action_sequence(next_path)
            
            # Check if stage is complete and handle stage transitions
            if self._is_stage_complete():
                if self._handle_stage_completion():
                    break  # Task completed
    
    def _load_program_info(self, rekep_program_dir):
        """Load program information and constraints"""
        # Load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        
        # Register keypoints
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        
        # Load constraints for all stages
        self.constraint_fns = self._load_constraints(rekep_program_dir)
    
    def _setup_task_environment(self):
        """Setup task environment variables"""
        # Keypoint movable mask - tracks which keypoints can be moved
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # End effector is always movable
        
        # Action queue for storing planned actions
        self.action_queue = []
    
    def _get_world_keypoints(self):
        """Get keypoints transformed to world coordinates"""
        scene_keypoints = self.env.get_keypoint_positions()
        print(f"Camera frame keypoints: {scene_keypoints}")
        
        world_keypoints = self._transform_keypoints_to_world(scene_keypoints)
        print(f"World frame keypoints: {world_keypoints}")
        
        return world_keypoints
    
    def _update_current_state(self, scene_keypoints):
        """Update current robot and environment state"""
        # Update keypoints with current end effector position
        self.keypoints = np.concatenate([[self._get_ee_position()], scene_keypoints], axis=0)
        
        # Update robot state
        self.curr_ee_pose = self._get_ee_pose()
        self.curr_joint_pos = self._get_joint_positions()
        
        # Update environment state
        self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
        self.collision_points = self.env.get_collision_points()
    
    def _generate_subgoal(self):
        """Generate next subgoal for current stage"""
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        
        print(f"Stage {self.stage}: Generating subgoal...")
        
        subgoal_pose, debug_dict = self.subgoal_solver.solve(
            self.curr_ee_pose,
            self.keypoints,
            self.keypoint_movable_mask,
            subgoal_constraints,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.is_grasp_stage,
            self.curr_joint_pos,
            from_scratch=self.first_iter
        )
        
        # Maintain current orientation
        subgoal_pose[3:7] = self.curr_ee_pose[3:7]
        
        # Apply grasp offset
        subgoal_pose = self._apply_grasp_offset(subgoal_pose)
        
        print(f"Next subgoal: {subgoal_pose}")
        print_opt_debug_dict(debug_dict)
        
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose, self.data_path)
        
        return subgoal_pose
    
    def _apply_grasp_offset(self, pose):
        """Apply offset for gripper positioning"""
        position = pose[:3]
        quat = pose[3:7]
        
        # Create rotation matrix and apply offset
        rotation_matrix = R.from_quat(quat).as_matrix()
        z_offset = np.array([0, 0, 0.16])  # 16cm offset along z-axis
        z_offset_world = rotation_matrix @ z_offset
        
        pose[:3] = position - z_offset_world
        return pose
    
    def _generate_path(self, subgoal):
        """Generate path to reach the subgoal"""
        path_constraints = self.constraint_fns[self.stage]['path']
        
        print(f"Generating path from {self.curr_ee_pose[:3]} to {subgoal[:3]}")
        
        path, debug_dict = self.path_solver.solve(
            self.curr_ee_pose,
            subgoal,
            self.keypoints,
            self.keypoint_movable_mask,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.curr_joint_pos,
            from_scratch=self.first_iter
        )
        
        self.first_iter = False
        print_opt_debug_dict(debug_dict)
        
        # Process path with interpolation
        processed_path = self._process_path(path)
        
        if self.visualize:
            self.visualizer.visualize_path(processed_path, self.data_path)
        
        return processed_path
    
    def _process_path(self, path):
        """Process path with spline interpolation"""
        # Combine current pose with planned path
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        
        # Calculate interpolation steps
        num_steps = get_linear_interpolation_steps(
            full_control_points[0], 
            full_control_points[-1],
            self.config['interpolate_pos_step_size'],
            self.config['interpolate_rot_step_size']
        )
        
        # Spline interpolation
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        
        # Create action sequence (6DOF pose + gripper)
        ee_action_seq = np.zeros((dense_path.shape[0], 7))
        ee_action_seq[:, :6] = dense_path[:, :6]
        ee_action_seq[:, 6] = self.env.get_gripper_null_action()
        
        return ee_action_seq
    
    def _execute_action_sequence(self, action_sequence):
        """Execute the planned action sequence"""
        self.action_queue = action_sequence.tolist()
        
        print(f"Executing {len(self.action_queue)} actions...")
        
        while len(self.action_queue) > 0:
            next_action = self.action_queue.pop(0)
            
            # Convert action format for robot execution
            processed_action = self._process_action_for_execution(next_action)
            
            # Execute action with precision flag for last action
            precise = len(self.action_queue) == 0
            self.robot_env.execute_action(processed_action, precise=precise)
    
    def _process_action_for_execution(self, action):
        """Convert action format for robot execution"""
        if len(action) == 7:  # [x, y, z, qx, qy, qz, qw]
            position = action[:3]
            quaternion = action[3:7]
            
            # Reorder quaternion from [qx, qy, qz, qw] to [qw, qx, qy, qz]
            quaternion = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
            
            # Convert quaternion to rotation vector
            rot_vec = quat_2_rotvec(quaternion)
            
            # Combine position and rotation vector
            processed_action = np.concatenate([position, rot_vec])
            print(f"Processed action: {processed_action}")
            
            return processed_action
        
        return action
    
    def _is_stage_complete(self):
        """Check if current stage is complete"""
        return len(self.action_queue) == 0
    
    def _handle_stage_completion(self):
        """Handle stage completion and transitions"""
        # Execute grasp/release actions
        if self.is_grasp_stage:
            self.robot_env._execute_grasp_action()
        elif self.is_release_stage:
            self.robot_env._execute_release_action()
        
        # Check if all stages are complete
        if self.stage == self.program_info['num_stages']:
            self._complete_task()
            return True
        
        # Move to next stage
        self._update_stage(self.stage + 1)
        return False
    
    def _complete_task(self):
        """Complete the task and return to initial position"""
        print(f"{bcolors.OKGREEN}Task completed! Returning to initial position...{bcolors.ENDC}")
        
        self.env.sleep(2.0)
        self._return_to_initial_position()
        self.robot_env._execute_release_action()
    
    def _update_stage(self, stage):
        """Update current stage and related flags"""
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][stage - 1] != -1
        
        # Ensure not both grasp and release stage
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        
        # Prepare gripper for grasp stage
        if self.is_grasp_stage:
            self.robot_env.robot.control_gripper(close=False)
        
        # Reset stage variables
        self.action_queue = []
        self._update_keypoint_movable_mask()
        self.first_iter = True
        
        print(f"Updated to stage {stage} - Grasp: {self.is_grasp_stage}, Release: {self.is_release_stage}")
    
    def _update_keypoint_movable_mask(self):
        """Update which keypoints can be moved in optimization"""
        for i in range(1, len(self.keypoint_movable_mask)):
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)
    
    def _return_to_initial_position(self):
        """Return robot to initial position after task completion"""
        if self.initial_position is not None:
            print(f"{bcolors.OKBLUE}Returning to initial position...{bcolors.ENDC}")
            self.robot_env.execute_action(self.initial_position, precise=False, speed=0.08)
            print(f"{bcolors.OKGREEN}Robot returned to initial position{bcolors.ENDC}")
        else:
            print(f"{bcolors.WARNING}No initial position stored, cannot return{bcolors.ENDC}")

    # Robot state getter methods
    def _get_joint_positions(self):
        """Get current joint positions"""
        return self.robot_env.robot.get_joint_positions()
    
    def _get_ee_position(self):
        """Get end effector position"""
        return self.robot_env.robot.get_tcp_pose()[:3]
    
    def _get_ee_pose(self):
        """Get end effector pose with quaternion"""
        ee_pos = self.robot_env.robot.get_tcp_pose()
        return pose2quat(ee_pos)

    # Constraint loading methods
    def _load_constraints(self, rekep_program_dir):
        """Load constraints for all stages"""
        constraint_fns = {}
        for stage in range(1, self.program_info['num_stages'] + 1):
            stage_dict = {}
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                if os.path.exists(load_path):
                    get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                    stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn)
                else:
                    stage_dict[constraint_type] = []
            constraint_fns[stage] = stage_dict
        return constraint_fns

    # Coordinate transformation methods
    def _transform_keypoints_to_world(self, keypoints):
        """Transform keypoints from camera coordinate system to world coordinate system"""
        keypoints = np.array(keypoints)
        
        # Load camera extrinsics
        ee2camera = self._load_camera_extrinsics()
        
        # Convert to homogeneous coordinates
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        
        # Get current end effector pose
        ee_pose = self._get_ee_pose()
        print(f"EE pose: {ee_pose}")
        
        quat = np.array([ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])  # [qx, qy, qz, qw]
        rotation = R.from_quat(quat).as_matrix()
        
        # Create transformation matrix
        base2ee = np.eye(4)
        base2ee[:3, :3] = rotation
        base2ee[:3, 3] = ee_pose[:3]
        
        # Camera frame transformation
        camera_frame = base2ee @ ee2camera
        
        # Apply transformation
        base_coords_homogeneous = (camera_frame @ keypoints_homogeneous.T).T
        
        # Convert to non-homogeneous coordinates
        return base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]

    def _load_camera_extrinsics(self):
        """Load camera extrinsics from YAML file"""
        extrinsics_path = 'cam_env/easy_handeye/easy_handeye_eye_on_hand.yaml'
        
        with open(extrinsics_path, 'r') as f:
            extrinsics_data = yaml.safe_load(f)
        
        # Extract transformation parameters
        t = extrinsics_data['transformation']
        quat = [t['qx'], t['qy'], t['qz'], t['qw']]  # [x, y, z, w] format
        pos = [t['x'], t['y'], t['z']]
        
        # Create 4x4 transformation matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R.from_quat(quat).as_matrix()
        extrinsics[:3, 3] = pos
        
        return extrinsics


def find_latest_rekep_directory():
    """Find the most recent rekep program directory"""
    vlm_query_dir = "./vlm_query/"
    
    if not os.path.exists(vlm_query_dir):
        return None
    
    vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                if os.path.isdir(os.path.join(vlm_query_dir, d))]
    
    return max(vlm_dirs, key=os.path.getmtime) if vlm_dirs else None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='UR5 robot manipulation controller')
    parser.add_argument('--instruction', type=str, help='Task instruction')
    parser.add_argument('--rekep_program_dir', type=str, help='ReKep program directory')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()

    # Find latest rekep directory if not provided
    rekep_dir = args.rekep_program_dir or find_latest_rekep_directory()
    
    if rekep_dir:
        print(f"\033[92mUsing ReKep directory: {rekep_dir}\033[0m")
    else:
        print("No ReKep directories found under vlm_query")
        sys.exit(1)

    # Create controller and execute task
    controller = UR5Controller(visualize=args.visualize)
    controller.execute_task(instruction=args.instruction, rekep_program_dir=rekep_dir)


if __name__ == "__main__":
    main()
