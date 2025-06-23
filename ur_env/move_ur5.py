import time
import asyncio
from vacuum_gripper import VacuumGripper
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rotations import rotvec_2_quat, quat_2_rotvec, pose2rotvec, pose2quat
import math

def get_robot_pose_and_joints(ip_address):
    """
    Connects to the UR5 robot using RTDE and retrieves its current pose in task space
    and joint space.

    :param ip_address: The IP address of the UR5 robot.
    """
    try:
        # Create instances of RTDE interfaces
        rtde_control = RTDEControlInterface(ip_address)
        rtde_receive = RTDEReceiveInterface(ip_address)

        # Get the task space pose (x, y, z, Rx, Ry, Rz)
        task_space_pose = rtde_receive.getActualTCPPose()
        task_space_pose= task_space_pose[:3] + [-val for val in task_space_pose[3:]]
        print("Task Space Pose (x, y, z, Rx, Ry, Rz):", task_space_pose)

        pos = pose2quat(task_space_pose)
        print("pos", pos)

        # Get the joint space positions (q1, q2, q3, q4, q5, q6)
        joint_space_positions = rtde_receive.getActualQ()
        print("Joint Space Positions (q1, q2, q3, q4, q5, q6):", joint_space_positions)

        return task_space_pose, joint_space_positions

    except Exception as e:
        print("Error while retrieving robot data:", e)
        return None, None

    finally:
        # Always stop RTDEControl to free resources
        rtde_control.stopScript()

def send_pose_to_robot(ip_address, pose, speed=0.05, acceleration=0.05):
    """
    Sends a pose to the robot controller to execute using moveL.

    :param ip_address: The IP address of the UR5 robot.
    :param pose: The pose to be executed (x, y, z, Rx, Ry, Rz).
    :param speed: Speed of the movement (default: 0.25 m/s).
    :param acceleration: Acceleration of the movement (default: 0.5 m/s^2).
    """
    try:
        rtde_control = RTDEControlInterface(ip_address)
        rtde_control.moveL(pose, speed, acceleration)
        print("Executed pose:", pose)

    except Exception as e:
        print("Error while sending pose to robot:", e)

    finally:
        rtde_control.stopScript()
        
def control_gripper(ip_address, width=0, speed=30, force=100, close=True):
    """
    Controls the Robotiq two-finger parallel gripper.

    :param ip_address: The IP address of the UR5 robot.
    :param width: Target width of the gripper (0 for fully closed, 255 for fully open).
    :param speed: Speed of the gripper movement (default: 30).
    :param force: Force applied by the gripper (default: 100).
    :param close: Whether to close (True) or open (False) the gripper.
    """
    async def execute_gripper_commands():
        gripper = VacuumGripper(ip_address)
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




if __name__ == "__main__":
    UR5_IP = "192.168.1.60"

    # close gripper
    control_gripper(UR5_IP, width=0, speed=30, force=100, close=False)
    time.sleep(0.5)
    # open gripper
    # control_gripper(UR5_IP, width=255, speed=100, force=100, close=True)
    # time.sleep(0.5)

    ## get pose and joint angle
    pose, joint = get_robot_pose_and_joints(UR5_IP)

    # send a psoe to robot controller
    try:
        # Example target pose to send to the robot
        target_pose = [-0.1096, -0.1042, 0.67, math.pi, 0.00, 0.00]
        send_pose_to_robot(UR5_IP, target_pose)

    except KeyboardInterrupt:
        print("Program interrupted.")