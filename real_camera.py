## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# 创建保存图片的目录
save_dir = "./data/realsense_captures"
os.makedirs(save_dir, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# --- Get Device Information ---
# Resolve the configuration to get the pipeline profile and device
print("Resolving pipeline configuration...")
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# --- Get and Print Device Info ---
device_product_line = str(device.get_info(rs.camera_info.product_line))
serial_number = str(device.get_info(rs.camera_info.serial_number))
print(f"Device Product Line: {device_product_line}")
print(f"Device Serial Number: {serial_number}")

# --- Configure Streams ---
# Try enabling the streams directly.
# The D405 might have different supported resolutions/formats for color.
try:
    print("Attempting to enable streams...")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    print("Streams enabled in config.")
except RuntimeError as e:
    print(f"Error enabling streams: {e}")
    print("The requested resolution/format might not be supported by the D405.")
    print("Common D405 color modes might be 480x270 or use YUYV format.")
    print("Check supported modes using RealSense Viewer or API.")
    exit(1)

# Start streaming
print("Starting pipeline...")
pipeline.start(config)
print("Pipeline started.")

# 获取相机内参
profile = pipeline.get_active_profile()

# 获取深度流内参
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

# 获取彩色流内参
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

# 打印内参信息
print("\n深度相机内参:")
print(f"分辨率: {depth_intrinsics.width}x{depth_intrinsics.height}")
print(f"焦距: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
print(f"主点: ppx={depth_intrinsics.ppx:.2f}, ppy={depth_intrinsics.ppy:.2f}")
print(f"畸变模型: {depth_intrinsics.model}")
print(f"畸变系数: {depth_intrinsics.coeffs}")

print("\n彩色相机内参:")
print(f"分辨率: {color_intrinsics.width}x{color_intrinsics.height}")
print(f"焦距: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
print(f"主点: ppx={color_intrinsics.ppx:.2f}, ppy={color_intrinsics.ppy:.2f}")
print(f"畸变模型: {color_intrinsics.model}")
print(f"畸变系数: {color_intrinsics.coeffs}")

try:
    frame_count = 0
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames(timeout_ms=5000)  # Added timeout
        if not frames:
            print("Timed out waiting for frames.")
            continue
            
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame_count += 1

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF  # Masking for cross-platform compatibility
        
        # ESC键退出
        if key == 27:
            print("ESC key pressed, exiting.")
            break
            
        # 按's'键保存图片
        elif key == ord('s'):
            # 生成时间戳作为文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 保存RGB图像
            color_path = os.path.join(save_dir, 'varied_camera_raw.png')
            cv2.imwrite(color_path, color_image)
            
            # 保存原始深度数据为npy格式
            depth_path = os.path.join(save_dir, 'varied_camera_depth.npy')
            np.save(depth_path, depth_image)  # 保存原始深度数据
            
            print(f'\n图片已保存到 {save_dir}:')
            print(f'- RGB图像: varied_camera_raw.png')
            print(f'- 深度数据: varied_camera_depth.npy')

except Exception as e:
    print(f"An error occurred during streaming: {e}")
    import traceback
    traceback.print_exc()  # Print detailed traceback for debugging

finally:
    # Stop streaming
    print("Stopping pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped and windows closed.")
