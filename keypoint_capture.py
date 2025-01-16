import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from keypoint_tracker import KeypointTracker
import os
import time

class KeypointCapture:
    def __init__(self):
        # 配置跟踪器
        self.keypoint_tracker = KeypointTracker(
            num_cam=1,  # 单相机
            boundaries={
                'x_lower': -0.5, 'x_upper': 0.5,  # 根据实际工作空间调整
                'y_lower': -0.5, 'y_upper': 0.5,
                'z_lower': 0.0, 'z_upper': 1.0
            },
            step=0.05  # 关键点采样步长
        )
        
        # 配置 RealSense
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()
        self.config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 对齐器
        self.align = rs.align(rs.stream.color)
        
        # 初始化标志
        self.is_initialized = False
        self.frame_count = 0
        
        # 存储上一帧数据
        self.prev_keypoints = None
        self.prev_features = None

    def start(self):
        # 启动相机
        self.profile = self.pipeline.start(self.config_rs)

    def stop(self):
        self.pipeline.stop()

    def get_frames(self):
        """获取对齐的RGB和深度图像"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 获取相机内参
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        return color_image, depth_image, depth_intrinsics

    def depth_to_points(self, depth_image, intrinsics):
        """将深度图转换为点云"""
        height, width = depth_image.shape
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # 展平数组
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_image.flatten()
        
        # 获取内参
        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy
        
        # 深度单位转换为米
        depth_scale = 0.001  # 通常RealSense深度单位是毫米
        depth_meters = depth_flat * depth_scale
        
        # 计算3D点
        x = (u_flat - ppx) * depth_meters / fx
        y = (v_flat - ppy) * depth_meters / fy
        z = depth_meters
        
        # 组合为点云
        points = np.vstack((x, y, z)).T
        points = points.reshape((height, width, 3))
        
        return points

    def process_frame(self):
        """处理单帧并检测关键点"""
        color_image, depth_image, intrinsics = self.get_frames()
        if color_image is None:
            return None
        
        # 转换为RGB格式
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # 设置相机参数
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        
        self.keypoint_tracker.set_intrinsics_extrinsics(
            intrinsics=[camera_matrix],  # 包装成list
            extrinsics=[np.eye(4)[:3]]   # 单相机使用单位矩阵
        )
        
        # 初始化关键点
        if not self.is_initialized:
            # 生成初始关键点网格
            x = np.arange(-0.4, 0.4, 0.1)
            y = np.arange(-0.4, 0.4, 0.1)
            z = np.array([0.3])  # 假设初始深度在0.3米
            X, Y, Z = np.meshgrid(x, y, z)
            initial_keypoints = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            
            # 初始化跟踪器
            self.keypoint_tracker.initialize(
                rgb_frames=[rgb],
                depth_frames=[depth_image],
                keypoints=initial_keypoints
            )
            self.is_initialized = True
            self.prev_keypoints = initial_keypoints
            return initial_keypoints, self.visualize_keypoints(color_image, initial_keypoints, camera_matrix)
        
        # 跟踪关键点
        tracked_keypoints = self.keypoint_tracker.track_keypoints(
            rgb_frames=[rgb],
            depth_frames=[depth_image],
            prev_keypoints=self.prev_keypoints
        )
        
        # 更新上一帧关键点
        self.prev_keypoints = tracked_keypoints
        
        # 可视化结果
        vis_image = self.visualize_keypoints(color_image, tracked_keypoints, camera_matrix)
        
        return tracked_keypoints, vis_image

    def visualize_keypoints(self, image, keypoints, camera_matrix):
        """可视化关键点"""
        vis_image = image.copy()
        
        # 将3D点投影到2D图像平面
        points_2d = []
        for point_3d in keypoints:
            x, y, z = point_3d
            if z > 0:  # 只处理相机前方的点
                pixel = np.dot(camera_matrix, np.array([x, y, z]))
                pixel = pixel / pixel[2]
                points_2d.append((int(pixel[0]), int(pixel[1])))
        
        # 绘制关键点
        for point in points_2d:
            cv2.circle(vis_image, point, 5, (0, 255, 0), -1)  # 绿色实心圆
            cv2.circle(vis_image, point, 7, (0, 0, 255), 2)   # 红色圆环
            
        return vis_image

def main():
    # 创建保存目录
    save_dir = "keypoint_captures"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化捕获器
    capturer = KeypointCapture()
    capturer.start()
    
    try:
        while True:
            # 处理帧
            result = capturer.process_frame()
            if result is None:
                continue
                
            keypoints, vis_image = result
            
            # 显示结果
            cv2.imshow('Keypoints', vis_image)
            
            if keypoints is not None:
                print(f'检测到的关键点数量: {len(keypoints)}')
            
            # 键盘控制
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
            elif key == ord('s'):  # 's'键保存当前帧
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # 保存图像
                cv2.imwrite(os.path.join(save_dir, f'keypoints_{timestamp}.png'), vis_image)
                # 保存关键点坐标
                np.save(os.path.join(save_dir, f'keypoints_{timestamp}.npy'), keypoints)
                print(f'已保存到 {save_dir}')
                
    finally:
        capturer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 