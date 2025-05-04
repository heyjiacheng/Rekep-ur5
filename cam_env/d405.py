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
serial_number = str(device.get_info(rs.camera_info.serial_number)) # <<<--- ADDED THIS LINE
print(f"Device Product Line: {device_product_line}")
print(f"Device Serial Number: {serial_number}") # <<<--- ADDED THIS LINE

# --- Removed the specific 'RGB Camera' check as it fails for D405 ---
# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)

# --- Configure Streams ---
# Try enabling the streams directly.
# The D405 might have different supported resolutions/formats for color.
# You might need to adjust these if you get errors later.
try:
    print("Attempting to enable streams...")
    # Using lower resolution often more compatible with D405 color derived from IR
    # You can try 640x480 first as in your original code
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Try bgr8 first, then maybe yuyv if needed
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

# --- Get Intrinsics ---
profile = pipeline.get_active_profile()

# Get depth stream intrinsics
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

# Get color stream intrinsics
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

# Print intrinsics information
print("\nDepth Camera Intrinsics:")
print(f"  Resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")
print(f"  Focal Length: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
print(f"  Principal Point: ppx={depth_intrinsics.ppx:.2f}, ppy={depth_intrinsics.ppy:.2f}")
print(f"  Distortion Model: {depth_intrinsics.model}")
print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")

print("\nColor Camera Intrinsics:")
print(f"  Resolution: {color_intrinsics.width}x{color_intrinsics.height}")
print(f"  Focal Length: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
print(f"  Principal Point: ppx={color_intrinsics.ppx:.2f}, ppy={color_intrinsics.ppy:.2f}")
print(f"  Distortion Model: {color_intrinsics.model}")
print(f"  Distortion Coefficients: {color_intrinsics.coeffs}")


# --- Main Loop ---
try:
    frame_count = 0
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames(timeout_ms=5000) # Added timeout
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

        # Keyboard control
        key = cv2.waitKey(1) & 0xFF # Masking for cross-platform compatibility

        # ESC key exit
        if key == 27:
            print("ESC key pressed, exiting.")
            break

        # 's' key save image
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            color_filename = f'color_{timestamp}.png'
            depth_filename = f'depth_{timestamp}.npy'
            color_path = os.path.join(save_dir, color_filename)
            depth_path = os.path.join(save_dir, depth_filename)

            cv2.imwrite(color_path, color_image)
            np.save(depth_path, depth_image)

            print(f'\nImages saved to {save_dir}:')
            print(f'- Color: {color_filename}')
            print(f'- Depth: {depth_filename}')

            # Optional: Save intrinsics here if needed (code from previous answer)

except Exception as e:
    print(f"An error occurred during streaming: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging

finally:
    # Stop streaming
    print("Stopping pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped and windows closed.")