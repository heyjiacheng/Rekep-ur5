#!/usr/bin/env python3
"""
Integrated Pipeline for Real-world Robot Manipulation
整合相机捕获、视觉处理和机器人动作执行的完整流程

执行顺序:
1. 相机捕获 - 从real_camera.py
2. 视觉处理 - 从real_vision.py  
3. 机器人动作 - 从ur5_action.py

同时保存末端执行器的所有pose数据
"""

import os
import time
import json
import numpy as np
import cv2
import torch
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# 相机相关导入
import pyrealsense2 as rs

# 视觉处理相关导入
from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.perception.gdino import GroundingDINO
from rekep.utils import bcolors, get_config

# 机器人控制相关导入
from ur_env.rotations import pose2quat, quat_2_rotvec
from rekep.environment import R2D2Env
from rekep.ik_solver import UR5IKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
from rekep.visualizer import Visualizer
from ur_env.ur5_env import RobotEnv
from rekep.utils import (
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")


class IntegratedRobotPipeline:
    """整合的机器人操作流水线"""
    
    def __init__(self, instruction=None, save_poses=True, execute_actions=True):
        """
        初始化整合流水线
        
        Args:
            instruction: 任务指令
            save_poses: 是否保存pose数据
            execute_actions: 是否执行真实机械臂动作
        """
        self.instruction = instruction or "Drop the box cutter into the blue box."
        self.save_poses = save_poses
        self.execute_actions = execute_actions
        self.save_dir = "./data/realsense_captures"
        self.poses_save_path = "./data/end_effector_poses.json"
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.poses_save_path), exist_ok=True)
        
        # 初始化组件
        self.pipeline = None
        self.vision_processor = None
        self.robot_controller = None
        self.pose_data = {
            "timestamp": datetime.now().isoformat(),
            "instruction": self.instruction,
            "poses": []
        }
        
        print(f"{bcolors.HEADER}=== 初始化整合机器人流水线 ==={bcolors.ENDC}")
        print(f"任务指令: {self.instruction}")
        print(f"执行动作: {self.execute_actions}")
        print(f"保存poses: {self.save_poses}")

    def setup_camera(self):
        """步骤1: 设置相机系统"""
        print(f"\n{bcolors.OKBLUE}=== 步骤1: 设置RealSense相机 ==={bcolors.ENDC}")
        
        # 配置深度和彩色流
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 指定相机序列号
        target_serial = "819612070593"
        config.enable_device(target_serial)
        
        # 检查RGB相机
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"相机序列号: {serial_number}")
        
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
                
        if not found_rgb:
            raise RuntimeError("需要带有彩色传感器的深度相机")
        
        # 配置流
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 开始流
        self.pipeline.start(config)
        
        # 获取相机内参
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.color_intrinsics = color_profile.get_intrinsics()
        
        print(f"深度相机内参: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
        print(f"彩色相机内参: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
        
    def capture_scene(self):
        """步骤2: 捕获场景图像"""
        print(f"\n{bcolors.OKBLUE}=== 步骤2: 捕获场景 ==={bcolors.ENDC}")
        
        if self.pipeline is None:
            raise RuntimeError("相机未初始化")
        
        time.sleep(2)  # 等待相机稳定
        # 等待帧
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("无法获取有效帧")
        
        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 保存图像
        color_path = os.path.join(self.save_dir, 'varied_camera_raw.png')
        depth_path = os.path.join(self.save_dir, 'varied_camera_depth.npy')
        
        cv2.imwrite(color_path, color_image)
        np.save(depth_path, depth_image)
        
        print(f"已保存图像到: {color_path}")
        print(f"已保存深度到: {depth_path}")
        
        return color_image, depth_image

    def setup_vision(self):
        """步骤3: 设置视觉处理系统"""
        print(f"\n{bcolors.OKBLUE}=== 步骤3: 设置视觉处理 ==={bcolors.ENDC}")
        
        global_config = get_config(config_path="./configs/config.yaml")
        config = global_config['main']
        
        # 设置随机种子
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        
        # 初始化组件
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        
        print("视觉处理系统初始化完成")

    def process_vision(self):
        """步骤4: 执行视觉处理"""
        print(f"\n{bcolors.OKBLUE}=== 步骤4: 视觉处理 ==={bcolors.ENDC}")
        
        color_path = os.path.join(self.save_dir, 'varied_camera_raw.png')
        depth_path = os.path.join(self.save_dir, 'varied_camera_depth.npy')
        
        # 加载图像
        bgr = cv2.imread(color_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        
        print(f"输入图像形状: {rgb.shape}")
        print(f"输入深度形状: {depth.shape}")
        
        # 目标检测
        print("执行Dino-X检测...")
        gdino = GroundingDINO()
        predictions = gdino.get_dinox(color_path)
        bboxes, masks = gdino.visualize_bbox_and_mask(predictions, color_path, './data/')
        masks = masks.astype(bool)
        masks = np.stack(masks, axis=0)
        
        print(f"生成了 {len(masks)} 个掩码")
        
        # 生成点云
        points = self.depth_to_pointcloud(depth)
        print(f"生成点云形状: {points.shape}")
        
        # 关键点提案和约束生成
        keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, masks)
        print(f'{bcolors.HEADER}获得 {len(keypoints)} 个提议关键点{bcolors.ENDC}')
        
        # 生成约束
        metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
        rekep_program_dir = self.constraint_generator.generate(projected_img, self.instruction, metadata)
        print(f'{bcolors.HEADER}约束已生成并保存在 {rekep_program_dir}{bcolors.ENDC}')
        
        return rekep_program_dir

    def depth_to_pointcloud(self, depth):
        """深度图转点云"""
        # D435默认内参
        class D435_Intrinsics:
            def __init__(self):
                self.fx = 616.57
                self.fy = 616.52
                self.ppx = 322.57
                self.ppy = 246.28
        
        intrinsics = D435_Intrinsics()
        depth_scale = 0.001
        
        height, width = depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        
        points = np.zeros((height * width, 3))
        valid_mask = depth > 0
        
        x = (u[valid_mask].flatten() - intrinsics.ppx) / intrinsics.fx
        y = (v[valid_mask].flatten() - intrinsics.ppy) / intrinsics.fy
        z = depth[valid_mask].flatten() * depth_scale
        
        x = np.multiply(x, z)
        y = np.multiply(y, z)
        
        valid_indices = np.where(valid_mask.flatten())[0]
        points[valid_indices] = np.stack((x, y, z), axis=-1)
        
        return points

    def setup_robot_controller(self):
        """步骤5: 设置机器人控制器"""
        print(f"\n{bcolors.OKBLUE}=== 步骤5: 设置机器人控制器 ==={bcolors.ENDC}")
        
        # 加载配置
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        
        # 设置随机种子
        seed = self.config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # 初始化机器人环境
        self.robot_env = RobotEnv()
        self.env = R2D2Env(global_config['env'])
        
        # 默认重置关节位置
        self.reset_joint_pos = np.array([
            -0.023413960133687794, -1.9976251761065882, 1.7851085662841797,
            4.942904949188232, -1.5486105124102991, -1.5801880995379847
        ])
        
        # IK求解器
        ik_solver = UR5IKSolver(
            reset_joint_pos=self.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        
        # 运动求解器
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.reset_joint_pos)
        
        print("机器人控制器初始化完成")

    def save_pose(self, pose, stage=None, action_type=None):
        """保存pose数据"""
        if not self.save_poses:
            return
            
        pose_entry = {
            "timestamp": time.time(),
            "stage": stage,
            "action_type": action_type,
            "pose": pose.tolist() if isinstance(pose, np.ndarray) else pose
        }
        self.pose_data["poses"].append(pose_entry)

    def execute_robot_task(self, rekep_program_dir):
        """步骤6: 执行机器人任务"""
        print(f"\n{bcolors.OKBLUE}=== 步骤6: 执行机器人任务 ==={bcolors.ENDC}")
        
        if rekep_program_dir is None:
            raise ValueError("需要提供rekep程序目录")
        
        # 保存初始位置
        self.initial_position = self.robot_env.robot.get_tcp_pose()
        self.save_pose(self.initial_position, stage=0, action_type="initial")
        
        # 加载程序信息
        self._load_program_info(rekep_program_dir)
        self._setup_task_environment()
        
        # 从阶段1开始
        self._update_stage(1)
        
        # 获取场景关键点并转换到世界坐标
        scene_keypoints = self._get_world_keypoints()
        
        # 主执行循环
        while True:
            # 更新当前状态
            self._update_current_state(scene_keypoints)
            
            # 生成下一个子目标和路径
            next_subgoal = self._generate_subgoal()
            next_path = self._generate_path(next_subgoal)
            
            # 执行计划路径
            self._execute_action_sequence(next_path)
            
            # 检查阶段是否完成并处理阶段转换
            if self._is_stage_complete():
                if self._handle_stage_completion():
                    break  # 任务完成

    def _load_program_info(self, rekep_program_dir):
        """加载程序信息和约束"""
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        self.constraint_fns = self._load_constraints(rekep_program_dir)

    def _setup_task_environment(self):
        """设置任务环境"""
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True
        self.action_queue = []

    def _get_world_keypoints(self):
        """获取转换到世界坐标的关键点"""
        scene_keypoints = self.env.get_keypoint_positions()
        print(f"相机坐标系关键点: {scene_keypoints}")
        
        world_keypoints = self._transform_keypoints_to_world(scene_keypoints)
        print(f"世界坐标系关键点: {world_keypoints}")
        
        return world_keypoints

    def _update_current_state(self, scene_keypoints):
        """更新当前机器人和环境状态"""
        self.keypoints = np.concatenate([[self._get_ee_position()], scene_keypoints], axis=0)
        self.curr_ee_pose = self._get_ee_pose()
        self.curr_joint_pos = self._get_joint_positions()
        self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
        self.collision_points = self.env.get_collision_points()
        
        # 保存当前pose
        self.save_pose(self.curr_ee_pose, stage=self.stage, action_type="current_state")

    def _generate_subgoal(self):
        """生成下一个子目标"""
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        
        print(f"阶段 {self.stage}: 生成子目标...")
        
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
        
        # 保持当前方向
        subgoal_pose[3:7] = self.curr_ee_pose[3:7]
        
        # 应用抓取偏移
        subgoal_pose = self._apply_grasp_offset(subgoal_pose)
        
        print(f"下一个子目标: {subgoal_pose}")
        print_opt_debug_dict(debug_dict)
        
        # 保存子目标pose
        self.save_pose(subgoal_pose, stage=self.stage, action_type="subgoal")
        
        return subgoal_pose

    def _apply_grasp_offset(self, pose):
        """应用抓取器偏移"""
        position = pose[:3]
        quat = pose[3:7]
        
        rotation_matrix = R.from_quat(quat).as_matrix()
        z_offset = np.array([0, 0, 0.16])
        z_offset_world = rotation_matrix @ z_offset
        
        pose[:3] = position - z_offset_world
        return pose

    def _generate_path(self, subgoal):
        """生成到达子目标的路径"""
        path_constraints = self.constraint_fns[self.stage]['path']
        
        print(f"生成路径从 {self.curr_ee_pose[:3]} 到 {subgoal[:3]}")
        
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
        
        processed_path = self._process_path(path)
        
        return processed_path

    def _process_path(self, path):
        """处理路径，进行样条插值"""
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        
        num_steps = get_linear_interpolation_steps(
            full_control_points[0],
            full_control_points[-1],
            self.config['interpolate_pos_step_size'],
            self.config['interpolate_rot_step_size']
        )
        
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        
        ee_action_seq = np.zeros((dense_path.shape[0], 7))
        ee_action_seq[:, :6] = dense_path[:, :6]
        ee_action_seq[:, 6] = self.env.get_gripper_null_action()
        
        return ee_action_seq

    def _execute_action_sequence(self, action_sequence):
        """执行计划的动作序列"""
        self.action_queue = action_sequence.tolist()
        
        print(f"执行 {len(self.action_queue)} 个动作...")
        
        while len(self.action_queue) > 0:
            next_action = self.action_queue.pop(0)
            
            processed_action = self._process_action_for_execution(next_action)
            
            # 保存每个执行的动作pose
            self.save_pose(processed_action, stage=self.stage, action_type="execution")
            
            # 只有在execute_actions为True时才执行真实动作
            if self.execute_actions:
                precise = len(self.action_queue) == 0
                self.robot_env.execute_action(processed_action, precise=precise)
            else:
                print(f"模拟执行动作: {processed_action[:3]} (位置)")  # 只打印位置部分

    def _process_action_for_execution(self, action):
        """转换动作格式以供机器人执行"""
        if len(action) == 7:
            position = action[:3]
            quaternion = action[3:7]
            
            quaternion = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
            rot_vec = quat_2_rotvec(quaternion)
            processed_action = np.concatenate([position, rot_vec])
            
            print(f"处理后的动作: {processed_action}")
            return processed_action
        
        return action

    def _is_stage_complete(self):
        """检查当前阶段是否完成"""
        return len(self.action_queue) == 0

    def _handle_stage_completion(self):
        """处理阶段完成和转换"""
        if self.is_grasp_stage:
            if self.execute_actions:
                self.robot_env._execute_grasp_action()
            else:
                print("模拟执行抓取动作")
            self.save_pose(self._get_ee_pose(), stage=self.stage, action_type="grasp")
        elif self.is_release_stage:
            if self.execute_actions:
                self.robot_env._execute_release_action()
            else:
                print("模拟执行释放动作")
            self.save_pose(self._get_ee_pose(), stage=self.stage, action_type="release")
        
        if self.stage == self.program_info['num_stages']:
            self._complete_task()
            return True
        
        self._update_stage(self.stage + 1)
        return False

    def _complete_task(self):
        """完成任务并返回初始位置"""
        print(f"{bcolors.OKGREEN}任务完成！返回初始位置...{bcolors.ENDC}")
        
        if self.execute_actions:
            self.env.sleep(2.0)
            self._return_to_initial_position()
            self.robot_env._execute_release_action()
        else:
            print("模拟返回初始位置和释放动作")
        
        # 保存最终pose
        final_pose = self._get_ee_pose()
        self.save_pose(final_pose, stage="final", action_type="task_complete")

    def _update_stage(self, stage):
        """更新当前阶段和相关标志"""
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][stage - 1] != -1
        
        assert self.is_grasp_stage + self.is_release_stage <= 1, "不能同时是抓取和释放阶段"
        
        if self.is_grasp_stage:
            if self.execute_actions:
                self.robot_env.robot.control_gripper(close=False)
            else:
                print("模拟打开夹具")
        
        self.action_queue = []
        self._update_keypoint_movable_mask()
        self.first_iter = True
        
        print(f"更新到阶段 {stage} - 抓取: {self.is_grasp_stage}, 释放: {self.is_release_stage}")

    def _update_keypoint_movable_mask(self):
        """更新优化中可移动的关键点"""
        for i in range(1, len(self.keypoint_movable_mask)):
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _return_to_initial_position(self):
        """任务完成后返回初始位置"""
        if self.initial_position is not None:
            print(f"{bcolors.OKBLUE}返回初始位置...{bcolors.ENDC}")
            if self.execute_actions:
                self.robot_env.execute_action(self.initial_position, precise=False, speed=0.08)
                print(f"{bcolors.OKGREEN}机器人已返回初始位置{bcolors.ENDC}")
            else:
                print("模拟返回初始位置")
            self.save_pose(self.initial_position, stage="return", action_type="return_home")
        else:
            print(f"{bcolors.WARNING}未存储初始位置，无法返回{bcolors.ENDC}")

    # 机器人状态获取方法
    def _get_joint_positions(self):
        """获取当前关节位置"""
        return self.robot_env.robot.get_joint_positions()

    def _get_ee_position(self):
        """获取末端执行器位置"""
        return self.robot_env.robot.get_tcp_pose()[:3]

    def _get_ee_pose(self):
        """获取末端执行器带四元数的位姿"""
        ee_pos = self.robot_env.robot.get_tcp_pose()
        return pose2quat(ee_pos)

    # 约束加载方法
    def _load_constraints(self, rekep_program_dir):
        """加载所有阶段的约束"""
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

    # 坐标变换方法
    def _transform_keypoints_to_world(self, keypoints):
        """将关键点从相机坐标系转换到世界坐标系"""
        keypoints = np.array(keypoints)
        
        ee2camera = self._load_camera_extrinsics()
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        
        ee_pose = self._get_ee_pose()
        print(f"EE位姿: {ee_pose}")
        
        quat = np.array([ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])
        rotation = R.from_quat(quat).as_matrix()
        
        base2ee = np.eye(4)
        base2ee[:3, :3] = rotation
        base2ee[:3, 3] = ee_pose[:3]
        
        camera_frame = base2ee @ ee2camera
        base_coords_homogeneous = (camera_frame @ keypoints_homogeneous.T).T
        
        return base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]

    def _load_camera_extrinsics(self):
        """从wrist_to_d435.tf加载相机外参"""
        extrinsics_path = 'cam_env/easy_handeye/wrist_to_d435.tf'
        with open(extrinsics_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        data_lines = lines[2:]
        translation = np.array([float(x) for x in data_lines[0].split()])
        rotation = np.array([[float(x) for x in line.split()] for line in data_lines[1:4]])
        
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation
        
        return extrinsics

    def cleanup(self):
        """清理资源"""
        print(f"\n{bcolors.OKBLUE}=== 清理资源 ==={bcolors.ENDC}")
        
        # 停止相机流
        if self.pipeline:
            self.pipeline.stop()
            print("相机流已停止")
        
        # 保存pose数据
        if self.save_poses and self.pose_data["poses"]:
            with open(self.poses_save_path, 'w') as f:
                json.dump(self.pose_data, f, indent=2)
            print(f"Pose数据已保存到: {self.poses_save_path}")
            print(f"总共保存了 {len(self.pose_data['poses'])} 个pose")

    def run_complete_pipeline(self):
        """运行完整的流水线"""
        try:
            print(f"{bcolors.HEADER}=== 开始完整机器人操作流水线 ==={bcolors.ENDC}")
            
            # 步骤1-2: 相机设置和场景捕获
            self.setup_camera()
            self.capture_scene()
            
            # 步骤3-4: 视觉处理
            self.setup_vision()
            rekep_program_dir = self.process_vision()
            
            # 步骤5-6: 机器人控制和执行
            self.setup_robot_controller()
            self.execute_robot_task(rekep_program_dir)
            
            print(f"{bcolors.OKGREEN}=== 流水线成功完成 ==={bcolors.ENDC}")
            
        except Exception as e:
            print(f"{bcolors.FAIL}流水线执行错误: {e}{bcolors.ENDC}")
            raise
        finally:
            self.cleanup()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='整合的机器人操作流水线')
    parser.add_argument('--instruction', type=str, 
                       default="Drop the box cutter into the blue box.",
                       help='任务指令')
    parser.add_argument('--save_poses', action='store_true', 
                       default=True, help='保存末端执行器pose')
    parser.add_argument('--execute_actions', action='store_true', 
                       default=False, help='执行真实机械臂动作（默认为模拟模式）')
    args = parser.parse_args()
    
    # 创建并运行流水线
    pipeline = IntegratedRobotPipeline(
        instruction=args.instruction,
        save_poses=args.save_poses,
        execute_actions=args.execute_actions
    )
    
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()