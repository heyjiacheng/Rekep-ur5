import pyrealsense2 as rs
import numpy as np
import cv2
import transform_utils as T
import os
import time


class RealCamera:
    """
    Defines the real camera class that mimics OGCamera interface
    """
    def __init__(self, width=640, height=480) -> None:
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        # Start streaming
        profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.color_intrinsics = color_profile.get_intrinsics()
        
        # Create align object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 构建内参矩阵 (与 OGCamera 一致, fx, fy, cx, cy)
        self.intrinsics = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # 构建外参矩阵，这里默认为单位阵，表示“世界坐标系”和“相机坐标系”暂时重合
        # 如果你在真实环境中有相机相对于“世界”的位姿，需要在这里更新 extrinsics
        self.extrinsics = np.eye(4)

    def get_params(self):
        """
        Get the intrinsic and extrinsic parameters of the camera
        """
        return {"intrinsics": self.intrinsics, "extrinsics": self.extrinsics}
    
    def get_obs(self):
        """
        Gets the image observation from the camera.
        Returns format matching OGCamera:
        {
            "rgb": (H, W, 3) RGB image
            "depth": (H, W) depth image in meters
            "points": (H, W, 3) 3D points in world coordinates
            "seg": (H, W) segmentation image (dummy all zeros)
            "intrinsic": (3, 3) camera intrinsics matrix
            "extrinsic": (4, 4) camera extrinsics matrix
        }
        """
        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None
            
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale  # 转为米
        color_image = np.asanyarray(color_frame.get_data())                     # BGR
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # 计算世界坐标系下的点云
        points_world = self._depth_to_world_points(depth_image)
        
        # 这里演示保存图像，可根据需求选择保留或删除
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = "camera_debug"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/color_{timestamp}.png", color_image)  # BGR 图保存
        depth_viz = (depth_image * 1000).astype(np.uint16)  # 转换到毫米单位以便直接可视化
        cv2.imwrite(f"{save_dir}/depth_{timestamp}.png", depth_viz)
        print("Saved images to:", save_dir)

        # 与 OGCamera 相同的输出格式
        ret = {
            "rgb": rgb_image,                  # (H, W, 3)
            "depth": depth_image,              # (H, W)
            "points": points_world,            # (H, W, 3) in world coordinates
            "seg": np.zeros_like(depth_image), # (H, W), dummy seg
            "intrinsic": self.intrinsics,      # (3, 3)
            "extrinsic": self.extrinsics       # (4, 4)
        }
        
        return ret

    def _depth_to_world_points(self, depth_image):
        """
        将深度图转换为世界坐标系下的三维点云。
        这里的实现参考了 OGCamera 里的 pixel_to_3d_points 做法：
        1. 根据内参将像素坐标转换为相机坐标
        2. 再使用外参（以及额外的坐标修正）转换到世界坐标系
        """
        H, W = depth_image.shape
        
        # 在图像平面上生成 (x, y) 网格
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        # 从内参矩阵中取出 fx, fy, cx, cy
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        
        # 将像素坐标转换为相机坐标
        z = depth_image
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        
        # 堆叠成 (H, W, 3)
        camera_coordinates = np.stack((x, y, z), axis=-1)
        
        # 变形到 (N, 3)
        camera_coordinates = camera_coordinates.reshape(-1, 3)
        # 转为齐次坐标 (N, 4)
        ones = np.ones((camera_coordinates.shape[0], 1))
        camera_coordinates_h = np.hstack((camera_coordinates, ones))
        
        # 由于在 Omniverse 中，Y, Z 轴有取反的约定，这里同样进行修正
        # (参考 OGCamera 的 pixel_to_3d_points 中的 T_mod)
        T_mod = np.array([
            [ 1.,  0.,  0.,  0.],
            [ 0., -1.,  0.,  0.],
            [ 0.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  1.]
        ])
        camera_coordinates_h = camera_coordinates_h @ T_mod
        
        # 将相机坐标转换到世界坐标：world = pose_inv(extrinsics) * cam
        # (如果 extrinsics 已经是“相机到世界”的变换，则这里应用它的逆)
        # 但如果 extrinsics 是“世界到相机”，则直接用 extrinsics @ camera_coordinates_h.T 即可
        # 下面假设 extrinsics 是“相机到世界”的变换矩阵，与 OGCamera 约定一致，需要先做 pose_inv。
        world_coordinates_h = T.pose_inv(self.extrinsics) @ camera_coordinates_h.T
        world_coordinates_h = world_coordinates_h.T
        
        # 转回普通坐标 (N, 3)
        world_coordinates = world_coordinates_h[:, :3] / world_coordinates_h[:, 3, np.newaxis]
        
        # reshape 回 (H, W, 3)
        world_coordinates = world_coordinates.reshape(H, W, 3)
        
        return world_coordinates
    
    def __del__(self):
        """Cleanup on deletion"""
        self.pipeline.stop()
