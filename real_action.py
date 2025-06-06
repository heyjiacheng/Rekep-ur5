import torch
import numpy as np
import json
import os
import sys
from scipy.spatial.transform import Rotation as R
import yaml
from ur_env.rotations import pose2quat
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


class RobotController:
    """Simplified robot controller"""
    
    def __init__(self, visualize=False):
        """Initialize robot controller"""
        self.config = self._load_config()
        self.visualize = visualize
        self._setup_components()
        self._set_random_seeds()
        
    def _load_config(self):
        """Load configuration file"""
        global_config = get_config(config_path="./configs/config.yaml")
        return global_config['main']
    
    def _setup_components(self):
        """Setup robot components"""
        # Robot environment
        self.robot_env = RobotEnv()
        self.env = R2D2Env(get_config(config_path="./configs/config.yaml")['env'])
        
        # Reset joint positions
        self.reset_joint_pos = np.array([
            0.19440510869026184, -1.9749982992755335, 1.5334253311157227, 
            5.154152870178223, -1.5606663862811487, 1.7688038349151611
        ])
        
        # IK solver
        ik_solver = UR5IKSolver(reset_joint_pos=self.reset_joint_pos, world2robot_homo=None)
        
        # Subgoal and path solvers
        global_config = get_config(config_path="./configs/config.yaml")
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
        """Execute robot task"""
        print(f"Executing task: {instruction}")
        
        if rekep_program_dir is None:
            print("Error: Need to provide rekep program directory")
            return
            
        self._run_stage(rekep_program_dir)

    def _run_stage(self, rekep_program_dir):
        """Run a single stage"""
        self.program_info = self._load_program_info(rekep_program_dir)
        self.constraint_fns = self._load_constraints(rekep_program_dir)
        
        stage = self._get_current_stage()
        self._setup_environment_state()
        self._update_stage(stage)
        
        actions = self._generate_actions()
        self._save_actions(actions, stage, rekep_program_dir)

    def _load_program_info(self, rekep_program_dir):
        """Load program information"""
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            return json.load(f)

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

    def _get_current_stage(self):
        """Get current stage from robot state"""
        try:
            with open('./robot_state.json', 'r') as f:
                robot_state = json.load(f)
                return robot_state.get('rekep_stage', 1)
        except FileNotFoundError:
            print("Warning: robot_state.json not found, using default stage 1")
            return 1

    def _setup_environment_state(self):
        """Setup environment state"""
        # Register keypoints
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        
        # Get scene information
        scene_keypoints = self.env.get_keypoint_positions()
        world_keypoints = self._transform_keypoints_to_world(scene_keypoints)
        print(f"World keypoints: {world_keypoints}")
        # Set state variables
        self.keypoints = np.concatenate([[self._get_ee_position()], world_keypoints], axis=0)
        self.curr_ee_pose = self._get_ee_pose()
        print(f"Current end effector pose: {self.curr_ee_pose}")
        self.curr_joint_pos = self._get_joint_positions()
        self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
        self.collision_points = self.env.get_collision_points()
        
        # Keypoint movable mask
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # End effector is always movable

    def _generate_actions(self):
        """Generate action sequence"""
        # Generate subgoal
        next_subgoal = self._generate_subgoal()
        print(f"Next subgoal: {next_subgoal}")
        
        # Generate path
        next_path = self._generate_path(next_subgoal)
        print(f"Generated path with {len(next_path)} points")
        
        return next_path

    def _generate_subgoal(self):
        """Generate subgoal"""
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        
        subgoal_pose, _ = self.subgoal_solver.solve(
            self.curr_ee_pose,
            self.keypoints,
            self.keypoint_movable_mask,
            subgoal_constraints,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.is_grasp_stage,
            self.curr_joint_pos,
            from_scratch=True
        )
        
        # Maintain current orientation
        subgoal_pose[3:7] = self.curr_ee_pose[3:7]
        
        # Apply grasp offset
        subgoal_pose = self._apply_grasp_offset(subgoal_pose)
        
        return subgoal_pose

    def _apply_grasp_offset(self, pose):
        """Apply grasp offset"""
        position = pose[:3]
        quat = pose[3:7]
        
        # Create rotation matrix and apply offset
        rotation_matrix = R.from_quat(quat).as_matrix()
        z_offset = np.array([0, 0, 0.16])  # Offset along z-axis
        z_offset_world = rotation_matrix @ z_offset
        
        pose[:3] = position - z_offset_world
        return pose

    def _generate_path(self, subgoal):
        """Generate path"""
        path_constraints = self.constraint_fns[self.stage]['path']
        
        path, _ = self.path_solver.solve(
            self.curr_ee_pose,
            subgoal,
            self.keypoints,
            self.keypoint_movable_mask,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.curr_joint_pos,
            from_scratch=True
        )
        
        return self._process_path(path)

    def _process_path(self, path):
        """Process path with interpolation"""
        # Combine current position and path
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
        
        # Create action sequence (7D pose + 1D gripper)
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        
        return ee_action_seq

    def _save_actions(self, actions, stage, rekep_program_dir):
        """Save action sequence"""
        # Save to output directory
        save_path = os.path.join('./outputs', 'action.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        action_data = {"ee_action_seq": actions.tolist(), "stage": stage}
        
        with open(save_path, 'w') as f:
            json.dump(action_data, f, indent=4)
        
        # Save to rekep program directory
        with open(os.path.join(rekep_program_dir, f'stage{stage}_actions.json'), 'w') as f:
            json.dump(action_data, f, indent=4)
        
        print(f"{bcolors.OKGREEN}Actions saved to {save_path}{bcolors.ENDC}")

    def _update_stage(self, stage):
        """Update stage state"""
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][stage - 1] != -1
        
        # Ensure cannot be both grasp and release stage
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        
        # If grasp stage, ensure gripper is open
        if self.is_grasp_stage:
            self.robot_env.robot.control_gripper(close=False)

    # Robot state getter methods
    def _get_joint_positions(self):
        """Get joint positions"""
        return self.robot_env.robot.get_joint_positions()
    
    def _get_ee_position(self):
        """Get end effector position"""
        return self.robot_env.robot.get_tcp_pose()[:3]
    
    def _get_ee_pose(self):
        """Get end effector pose"""
        ee_pos = self.robot_env.robot.get_tcp_pose()
        return pose2quat(ee_pos)

    # Coordinate transformation methods
    def _transform_keypoints_to_world(self, keypoints):
        """Transform keypoints from camera coordinate system to world coordinate system"""
        keypoints = np.array(keypoints)
        
        # Load camera extrinsics (now base2camera)
        base2camera = self._load_camera_extrinsics()
        
        # Convert to homogeneous coordinates
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        
        # Apply transformation directly (camera to world/base)
        base_coords_homogeneous = (base2camera @ keypoints_homogeneous.T).T
        
        # Convert to non-homogeneous coordinates
        return base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]

    def _load_camera_extrinsics(self):
        """Load camera extrinsics"""
        extrinsics_path = 'cam_env/easy_handeye/easy_handeye_eye_on_hand.yaml'
        
        with open(extrinsics_path, 'r') as f:
            extrinsics_data = yaml.safe_load(f)
        
        # Extract quaternion (YAML format: [w,x,y,z] -> Scipy format: [x,y,z,w])
        qw = extrinsics_data['transformation']['qw']
        qx = extrinsics_data['transformation']['qx']
        qy = extrinsics_data['transformation']['qy']
        qz = extrinsics_data['transformation']['qz']
        
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Extract translation
        tx = extrinsics_data['transformation']['x']
        ty = extrinsics_data['transformation']['y']
        tz = extrinsics_data['transformation']['z']
        
        # Create 4x4 transformation matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot
        extrinsics[:3, 3] = [tx, ty, tz]
        
        return extrinsics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simplified robot controller')
    parser.add_argument('--instruction', type=str, help='Task instruction')
    parser.add_argument('--rekep_program_dir', type=str, help='ReKep program directory')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()

    # Find latest VLM query directory
    vlm_query_dir = "./vlm_query/"
    vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                if os.path.isdir(os.path.join(vlm_query_dir, d))]
    
    if vlm_dirs:
        newest_rekep_dir = max(vlm_dirs, key=os.path.getmtime)
        print(f"\033[92mUsing latest directory: {newest_rekep_dir}\033[0m")
    else:
        print("No directories found under vlm_query")
        sys.exit(1)

    # Create controller and execute task
    controller = RobotController(visualize=args.visualize)
    controller.execute_task(instruction=args.instruction, rekep_program_dir=newest_rekep_dir)


if __name__ == "__main__":
    main()
