import asyncio
from enum import Enum
from typing import Union
from ur_env.vacuum_gripper import VacuumGripper
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from ur_env.rotations import rotvec_2_quat, quat_2_rotvec, pose2rotvec, pose2quat
import time

class RobotEnv:
    def __init__(self, ip='192.168.1.60'):
        # 初始化真实机械臂环境   
        self.ip = ip

        # 初始化机械臂
        self.robot = ur5Robot(ip)
        
    def execute_action(self, action, precise=False, speed=0.08, acceleration=0.08):
        """
        Execute a single action with pose change only (without gripper control)
        
        Args:
            action: List containing position and orientation [x, y, z, rx, ry, rz]
            precise: Whether to execute precisely (usually the last action in the queue)
            speed: Movement speed (default: 0.05 m/s)
            acceleration: Movement acceleration (default: 0.05 m/s^2)
            position_only: If True, only change position and keep current orientation
        
        Returns:
            bool: Whether the execution was successful
        """
        try:            
            # Use reduced speed and acceleration for precise movements
            if precise:
                speed = speed / 2
                acceleration = acceleration / 2
                print("Performing precise action")
            
            # Execute pose change
            print(f"Moving to pose: {action}")
            self.robot.send_pose_to_robot(action, speed, acceleration)
            
            # Allow time for movement to complete
            self.sleep(0.5)  # Adjust based on movement size
            
            return True
            
        except Exception as e:
            print(f"Error performing action: {e}")
            return False
            
    def _execute_grasp_action(self):
        """
        Execute a grasp action (close gripper)
        
        Returns:
            bool: Whether the grasp was successful
        """
        try:
            print("Executing grasp action")
            self.robot.control_gripper(close=True)
            self.sleep(1.5)  # Wait for gripper to close
            print("Gripper closed")
            return True
        except Exception as e:
            print(f"Error executing grasp action: {e}")
            return False
            
    def _execute_release_action(self):
        """
        Execute a release action (open gripper)
        
        Returns:
            bool: Whether the release was successful
        """
        try:
            print("Executing release action")
            self.robot.control_gripper(close=False)
            self.sleep(1.5)  # Wait for gripper to open
            print("Gripper opened")
            return True
        except Exception as e:
            print(f"Error executing release action: {e}")
            return False
    
    def sleep(self, seconds):
        """Wait for specified duration"""
        time.sleep(seconds)

class ur5Robot:
    def __init__(self, ip='192.168.1.60'):
        # 连接机器人
        self.ip = ip

    def get_robot_pose_and_joints(self):
        """
        Connects to the UR5 robot using RTDE and retrieves its current pose in task space
        and joint space.
        """
        try:
            # Create instances of RTDE interfaces
            rtde_control = RTDEControlInterface(self.ip)
            rtde_receive = RTDEReceiveInterface(self.ip)

            # Get the task space pose (x, y, z, Rx, Ry, Rz)
            task_space_pose = rtde_receive.getActualTCPPose()
            task_space_pose = task_space_pose[:3] + [-val for val in task_space_pose[3:]]
            # print("Task Space Pose (x, y, z, Rx, Ry, Rz):", task_space_pose)

            pos = pose2quat(task_space_pose)
            # print("pos", pos)

            # Get the joint space positions (q1, q2, q3, q4, q5, q6)
            joint_space_positions = rtde_receive.getActualQ()
            # print("Joint Space Positions (q1, q2, q3, q4, q5, q6):", joint_space_positions)

            return task_space_pose, joint_space_positions

        except Exception as e:
            print("Error while retrieving robot data:", e)
            return None, None

        finally:
            # Always stop RTDEControl to free resources
            rtde_control.stopScript()

    def send_pose_to_robot(self, pose, speed=0.05, acceleration=0.05):
        """
        Sends a pose to the robot controller to execute using moveL.

        :param pose: The pose to be executed (x, y, z, Rx, Ry, Rz).
        :param speed: Speed of the movement (default: 0.05 m/s).
        :param acceleration: Acceleration of the movement (default: 0.05 m/s^2).
        """
        try:
            rtde_control = RTDEControlInterface(self.ip)
            rtde_control.moveL(pose, speed, acceleration)
            print("Executed pose:", pose)

        except Exception as e:
            print("Error while sending pose to robot:", e)

        finally:
            rtde_control.stopScript()

    def send_joints_to_robot(self, joint_positions, speed=1.0, acceleration=1.4):
        """
        Sends joint angles to the robot controller to execute using moveJ.

        :param joint_positions: The joint positions to be executed (q1, q2, q3, q4, q5, q6).
        :param speed: Speed of the movement (default: 1.0 rad/s).
        :param acceleration: Acceleration of the movement (default: 1.4 rad/s^2).
        """
        try:
            rtde_control = RTDEControlInterface(self.ip)
            rtde_control.moveJ(joint_positions, speed, acceleration)
            print("Executed joint positions:", joint_positions)

        except Exception as e:
            print("Error while sending joint positions to robot:", e)

        finally:
            rtde_control.stopScript()

    def control_gripper(self, width=0, speed=30, force=100, close=True):
        """
        Controls the Robotiq two-finger parallel gripper.

        :param width: Target width of the gripper (0 for fully closed, 255 for fully open).
        :param speed: Speed of the gripper movement (default: 30).
        :param force: Force applied by the gripper (default: 100).
        :param close: Whether to close (True) or open (False) the gripper.
        """
        async def execute_gripper_commands():
            gripper = VacuumGripper(self.ip)
            await gripper.connect()
            await gripper.activate()
            if close:
                await gripper.close_gripper(force=force, speed=speed)
                print("Gripper closed.")
            else:
                await gripper.open_gripper(force=force, speed=speed)
                print("Gripper opened.")
            await gripper.disconnect()

        try:
            # Get the current running event loop or create a new one
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the asynchronous function in the loop
            if loop.is_running():
                asyncio.ensure_future(execute_gripper_commands())
            else:
                loop.run_until_complete(execute_gripper_commands())

        except Exception as e:
            print("Error controlling gripper:", e)

    def get_tcp_pose(self):
        """
        获取当前末端执行器位姿
        Returns:
            list: 当前TCP位姿 [x, y, z, rx, ry, rz]
        """
        pose, _ = self.get_robot_pose_and_joints()
        return pose
    
    def get_joint_positions(self):
        """
        获取当前关节位置
        """
        _, joint_positions = self.get_robot_pose_and_joints()
        return joint_positions