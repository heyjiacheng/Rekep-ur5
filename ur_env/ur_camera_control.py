import time
import numpy as np
import cv2
import os
import pyrealsense2 as rs
import math
# Import functions from move_ur5.py
from move_ur5 import send_pose_to_robot, get_robot_pose_and_joints, control_gripper
from scipy.spatial.transform import Rotation # For easier rotation handling

# === Configuration ===
UR5_IP = "192.168.1.60"
SAVE_DIR = "./data/images"  # Directory to save images
OBJECT_POS = np.array([-0.40123, -0.10366, 0.1])  # Approximate object center [x, y, z]
CAMERA_HEIGHT = 0.6700  # Constant height for the camera (robot TCP z-coordinate)
CIRCLE_RADIUS = 0.20  # Radius of the circle around the object (in meters)
NUM_IMAGES = 15  # Number of images to capture over arc
ARC_DEGREES = 90.0 # Angle of the arc to cover
TILT_ANGLE_DEG = 15.0 # Inward tilt angle for the camera (degrees)
INITIAL_POSE = [-0.4196, -0.1042, CAMERA_HEIGHT, math.pi, 0.00, 0.00] # Start pose - adjust Z to CAMERA_HEIGHT

# Camera offset relative to TCP (flange)
# x: 0.03003, y: -0.12303, z: -0.03641 (User updated)
CAMERA_OFFSET_TCP = np.array([0.03003, -0.12303, -0.03641])
# ===================

def calculate_target_poses(object_pos, height, radius, num_poses, arc_degrees, tilt_degrees, camera_offset):
    """
    Calculates TCP poses in an arc around the object,
    with the camera looking at the object and tilted inwards.
    Args:
        object_pos (np.array): Target object position [x, y, z].
        height (float): Desired camera height (z-coordinate).
        radius (float): Radius of the arc around the object.
        num_poses (int): Number of poses to generate.
        arc_degrees (float): The total angle of the arc to cover.
        tilt_degrees (float): Desired inward camera tilt in degrees.
        camera_offset (np.array): Camera position relative to TCP [x, y, z].
    Returns:
        list: List of calculated TCP poses [[x, y, z, rx, ry, rz], ...].
    """
    tcp_poses = []
    world_up = np.array([0.0, 0.0, 1.0])
    arc_radians = math.radians(arc_degrees)
    tilt_radians = math.radians(tilt_degrees)

    # Calculate start angle (e.g., start from -arc_radians/2 relative to object X-axis)
    start_angle_rad = -arc_radians / 2.0
    
    for i in range(num_poses):
        # Calculate angle for this pose within the arc
        if num_poses > 1:
            angle_rad = start_angle_rad + arc_radians * (i / (num_poses - 1))
        else:
            angle_rad = start_angle_rad

        # 1. Calculate desired CAMERA position on the arc
        cam_x = object_pos[0] + radius * math.cos(angle_rad)
        cam_y = object_pos[1] + radius * math.sin(angle_rad)
        cam_z = height
        desired_cam_pos = np.array([cam_x, cam_y, cam_z])
        
        # 2. Calculate base CAMERA orientation (no tilt) looking at the object
        # Z-axis points from camera towards object
        cam_z_axis_no_tilt = object_pos - desired_cam_pos 
        norm_z = np.linalg.norm(cam_z_axis_no_tilt)
        if norm_z < 1e-6:
             print(f"Warning: Camera position nearly coincident with object position at step {i}.")
             cam_z_axis_no_tilt = -world_up
        else:
            cam_z_axis_no_tilt = cam_z_axis_no_tilt / norm_z
            
        # X-axis is horizontal, perpendicular to Z and world Up
        cam_x_axis = np.cross(world_up, cam_z_axis_no_tilt)
        norm_x = np.linalg.norm(cam_x_axis)
        if norm_x < 1e-6: 
            if np.dot(cam_z_axis_no_tilt, world_up) < -0.999: # Looking straight down
                cam_x_axis = np.array([1.0, 0.0, 0.0])
            elif np.dot(cam_z_axis_no_tilt, world_up) > 0.999: # Looking straight up
                 cam_x_axis = np.array([-1.0, 0.0, 0.0])
            else: # Should not happen, fallback
                cam_x_axis = np.array([1.0, 0.0, 0.0]) 
        else:
             cam_x_axis = cam_x_axis / norm_x
        
        # Y-axis completes the right-handed system (camera's 'up' without tilt)
        cam_y_axis_no_tilt = np.cross(cam_z_axis_no_tilt, cam_x_axis)

        # 3. Apply the inward tilt rotation
        # Rotation is around the camera's X-axis (cam_x_axis)
        tilt_rotation = Rotation.from_rotvec(tilt_radians * cam_x_axis)
        
        # Apply tilt to the base orientation axes
        final_cam_z_axis = tilt_rotation.apply(cam_z_axis_no_tilt)
        final_cam_y_axis = tilt_rotation.apply(cam_y_axis_no_tilt)
        # final_cam_x_axis remains cam_x_axis as it's the rotation axis
        final_cam_x_axis = cam_x_axis
        
        # Create the final tilted CAMERA rotation matrix
        cam_rotation_matrix = np.array([final_cam_x_axis, final_cam_y_axis, final_cam_z_axis]).T

        # 4. Calculate required TCP position based on final camera pose
        required_tcp_pos = desired_cam_pos - cam_rotation_matrix @ camera_offset
        
        # 5. Convert final CAMERA rotation matrix to rotation vector for the TCP
        r = Rotation.from_matrix(cam_rotation_matrix)
        tcp_rotvec = r.as_rotvec()
        
        # 6. Combine required TCP position and rotation vector
        tcp_pose = list(required_tcp_pos) + list(tcp_rotvec)
        tcp_poses.append(tcp_pose)
        
    return tcp_poses

def initialize_camera():
    """
    Initialize and configure the RealSense camera.
    
    :return: pipeline, color_intrinsics, depth_intrinsics
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        return None, None, None

    # We still need to enable depth stream for the camera to work properly
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Get camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()

    # Print camera intrinsics to terminal
    print("\n==== 相机内参 ====")
    # print("\n深度相机内参:") # Depth intrinsics not needed for user
    # print(f"分辨率: {depth_intrinsics.width}x{depth_intrinsics.height}")
    # print(f"焦距: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
    # print(f"主点: ppx={depth_intrinsics.ppx:.2f}, ppy={depth_intrinsics.ppy:.2f}")
    # print(f"畸变模型: {depth_intrinsics.model}")
    # print(f"畸变系数: {depth_intrinsics.coeffs}")

    print("\n彩色相机内参:")
    print(f"分辨率: {color_intrinsics.width}x{color_intrinsics.height}")
    print(f"焦距: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
    print(f"主点: ppx={color_intrinsics.ppx:.2f}, ppy={color_intrinsics.ppy:.2f}")
    print(f"畸变模型: {color_intrinsics.model}")
    print(f"畸变系数: {color_intrinsics.coeffs}")
    print("\n================")

    return pipeline, color_intrinsics, depth_intrinsics

def capture_rgb_image(pipeline, save_dir, pose_str):
    """
    Capture and save only RGB image from the RealSense camera.
    
    :param pipeline: RealSense pipeline
    :param save_dir: Directory to save images
    :param pose_str: String representation of the robot pose (already formatted, joined by '_')
    :return: True if capture successful, False otherwise
    """
    # Allow camera to stabilize
    for _ in range(5):
        frames = pipeline.wait_for_frames()
        
    # Wait for frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        print("Error: Failed to get color frame.")
        return False

    # Convert image to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Generate filename with pose info
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Use the already formatted pose_str directly
    filename_base = f"pose_{pose_str}_{timestamp}"

    # Save RGB image as JPG
    color_path = os.path.join(save_dir, f'{filename_base}.jpg')
    cv2.imwrite(color_path, color_image)
    
    print(f'RGB image saved to {save_dir}:')
    print(f'- {filename_base}.jpg')
    
    return True

def main():
    # Ensure INITIAL_POSE uses the configured CAMERA_HEIGHT
    initial_pose_actual = list(INITIAL_POSE[:2]) + [CAMERA_HEIGHT] + list(INITIAL_POSE[3:])
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("初始化相机中...")
    pipeline, color_intrinsics, depth_intrinsics = initialize_camera()
    if pipeline is None:
        print("相机初始化失败，退出程序")
        return
    
    print(f"\n计算目标 TCP 位姿 (考虑相机偏移, {ARC_DEGREES}度弧线, {TILT_ANGLE_DEG}度内倾)...")
    target_tcp_poses = calculate_target_poses(OBJECT_POS, CAMERA_HEIGHT, CIRCLE_RADIUS, 
                                            NUM_IMAGES, ARC_DEGREES, TILT_ANGLE_DEG, CAMERA_OFFSET_TCP)
    print(f"已生成 {len(target_tcp_poses)} 个目标 TCP 位姿.")
    
    print("\n程序启动成功！")
    print("控制说明:")
    print(f"- 程序将移动到初始位姿，然后按顺序移动到 {NUM_IMAGES} 个环绕物体 {ARC_DEGREES} 度、内倾 {TILT_ANGLE_DEG} 度的位姿")
    print("- 机器人到达位姿后，按 's' 拍照")
    print("- 按 'ESC' 跳过当前位姿")
    print("- 按 'q' 退出程序")
    
    robot_control = None # Initialize variable for finally block
    try:
        # Establish connection once if possible (depends on move_ur5 implementation)
        # If send_pose_to_robot connects/disconnects each time, this is not needed
        # robot_control = RTDEControlInterface(UR5_IP) # Example if connection persistence is desired
        
        # 1. Move to initial pose
        print(f"\n移动到初始位姿: {[f'{v:.4f}' for v in initial_pose_actual]}")
        send_pose_to_robot(UR5_IP, initial_pose_actual, speed=0.1, acceleration=0.2)
        print("已到达初始位姿. 准备开始采集...")
        time.sleep(1) # Pause briefly

        # 2. Loop through target TCP poses
        for pose_idx, target_tcp_pose in enumerate(target_tcp_poses):
            print(f"\n[{pose_idx+1}/{NUM_IMAGES}] 移动机器人 TCP 到位姿: {[f'{v:.4f}' for v in target_tcp_pose]}")
            
            # 机器人移动到目标位姿
            success = send_pose_to_robot(UR5_IP, target_tcp_pose, speed=0.08, acceleration=0.15)
            # if not success:
            #     print(f"警告: 移动到目标位姿 {pose_idx+1} 失败，可能无法到达。跳过此位姿。")
            #     continue # Skip to next pose if move failed
            
            # 生成位姿字符串用于文件命名 (使用 TCP 位姿)
            pose_str_for_filename = "_".join([f"{val:.4f}" for val in target_tcp_pose])
            
            # 显示摄像头画面并等待确认拍照
            print(f"机器人已到达位姿 {pose_idx+1}，请按 's' 拍照，'ESC' 跳过，'q' 退出程序")
            
            capture_confirmed = False
            while not capture_confirmed:
                # 获取帧
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    # print(".", end="") # Optional: indicate frame drop
                    continue
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                
                # 显示图像
                cv2.namedWindow('RealSense RGB', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense RGB', color_image)
                
                key = cv2.waitKey(1)
                
                # 'q'键退出整个程序
                if key == ord('q'):
                    print("用户请求退出程序")
                    raise SystemExit("用户退出") 
                
                # ESC键跳过当前位姿
                elif key == 27:
                    print(f"已跳过位姿 {pose_idx+1}")
                    capture_confirmed = True 
                
                # 's'键拍照
                elif key == ord('s'):
                    if capture_rgb_image(pipeline, SAVE_DIR, pose_str_for_filename):
                         capture_confirmed = True 
                    else:
                        print("拍照失败，请重试或跳过 ('ESC')")
        
        print(f"\n所有 {NUM_IMAGES} 个位姿已处理完成！")
    
    except SystemExit as e:
        print(e)
    except Exception as e:
        print(f"发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        print("\n清理资源并退出程序...")
        if pipeline:
            pipeline.stop()
            print("RealSense pipeline stopped.")
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")
        
        # Attempt to stop any running robot script
        try:
             # Make sure RTDEControlInterface is imported if not done above
             from rtde_control import RTDEControlInterface 
             print(f"正在尝试停止 {UR5_IP} 上的机器人脚本...")
             robot_control = RTDEControlInterface(UR5_IP) #, connect_timeout=1.0) # Short timeout
             if robot_control.isConnected():
                 robot_control.stopScript()
                 print("机器人脚本已停止。")
             else:
                 print("无法连接到机器人以停止脚本 (可能已停止或无法访问)。")
             # Explicitly disconnect if connection was made here
             if robot_control and robot_control.isConnected():
                 robot_control.disconnect()

        except ImportError:
             print("无法导入 RTDEControlInterface，跳过机器人脚本停止步骤。")
        except Exception as e:
             print(f"尝试停止机器人脚本时出错 (可能已停止): {e}")
        
        print("程序已退出")

if __name__ == "__main__":
    main() 