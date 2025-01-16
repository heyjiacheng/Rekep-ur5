from d3fields.fusion import Fusion
import numpy as np
import torch

class KeypointTracker:
    def __init__(self, num_cam, boundaries, step, feat_backbone='dinov2'):
        """
        初始化 KeypointTracker，用于关键点跟踪任务。

        Args:
            num_cam (int): 相机数量。
            boundaries (dict): 三维描述符场的边界，格式为 {'x_lower': ..., 'x_upper': ..., ...}。
            step (float): 三维网格的步长，决定描述符场的分辨率。
            feat_backbone (str): 特征提取骨干网络（默认为 'dinov2'）。
        """
        self.boundaries = boundaries  # 描述符场边界
        self.step = step  # 三维网格步长
        self.fusion = Fusion(num_cam=num_cam, feat_backbone=feat_backbone)  # 初始化 Fusion 实例
        self.target_features = None  # 上一帧的关键点特征
        self.intrinsics = None  # 相机内参 (num_cam, 3, 3)
        self.extrinsics = None  # 相机外参 (num_cam, 3, 4)

    def set_intrinsics_extrinsics(self, intrinsics, extrinsics):
        """
        设置相机的内参和外参。

        Args:
            intrinsics (np.ndarray): 相机内参矩阵 (num_cam, 3, 3)。
            extrinsics (np.ndarray): 相机外参矩阵 (num_cam, 3, 4)。
        """
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def initialize(self, rgb_frames, depth_frames, keypoints):
        """
        初始化描述符场并提取初始关键点的特征。

        Args:
            rgb_frames (list of np.ndarray): 多视角 RGB 图像列表，每个形状为 (H, W, 3)。
            depth_frames (list of np.ndarray): 多视角深度图像列表，每个形状为 (H, W)。
            keypoints (np.ndarray): 初始关键点位置，形状为 (N, 3)。
        """
        # 构建 obs 输入
        colors = np.stack(rgb_frames, axis=0)  # (num_cam, H, W, 3)
        depths = np.stack([depth / 1000.0 for depth in depth_frames], axis=0)  # (num_cam, H, W)
        obs = {
            'color': colors,
            'depth': depths,
            'pose': self.extrinsics,
            'K': self.intrinsics,
        }

        # 更新 Fusion，初始化描述符场
        self.fusion.update(obs)

        # 提取初始关键点的特征
        keypoints_tensor = torch.from_numpy(keypoints).to(device='cuda', dtype=torch.float32)
        with torch.no_grad():
            self.target_features = self.fusion.batch_eval(keypoints_tensor, return_names=['dino_feats'])

    def get_keypoint_positions(self, keypoints, rgb_frames, depth_frames):
        """
        更新当前帧的关键点位置。

        Args:
            keypoints (np.ndarray): 当前帧输入的关键点位置，形状为 (N, 3)。
            rgb_frames (list of np.ndarray): 多视角 RGB 图像列表，每个形状为 (H, W, 3)。
            depth_frames (list of np.ndarray): 多视角深度图像列表，每个形状为 (H, W)。

        Returns:
            np.ndarray: 更新后的关键点位置，形状为 (N, 3)。
        """
        # 构建 obs 输入
        colors = np.stack(rgb_frames, axis=0)  # (num_cam, H, W, 3)
        depths = np.stack([depth / 1000.0 for depth in depth_frames], axis=0)  # (num_cam, H, W)
        obs = {
            'color': colors,
            'depth': depths,
            'pose': self.extrinsics,
            'K': self.intrinsics,
        }

        # 更新 Fusion 描述符场
        self.fusion.update(obs)

        # 将传入的关键点转换为 Tensor
        keypoints_tensor = torch.from_numpy(keypoints).to(device='cuda', dtype=torch.float32)
        keypoints_tensor.requires_grad = True

        # 梯度优化关键点位置
        optimizer = torch.optim.Adam([keypoints_tensor], lr=1e-3)

        for _ in range(10):  # 优化迭代 10 次
            optimizer.zero_grad()
            current_features = self.fusion.batch_eval(keypoints_tensor, return_names=['dino_feats'])
            loss = torch.nn.functional.mse_loss(current_features, self.target_features)
            loss.backward()
            optimizer.step()

        # 更新关键点位置
        updated_keypoints = keypoints_tensor.detach().cpu().numpy()
        return updated_keypoints
