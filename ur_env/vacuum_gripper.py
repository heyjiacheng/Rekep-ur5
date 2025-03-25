"""
MIT License

Copyright (c) 2019 Anders Prier Lindvig - SDU Robotics
Copyright (c) 2020 Fabian Freihube - DavinciKitchen GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Module to control Robotiq's gripper 2F-85 and Hand-E.
Originally from here: https://sdurobotics.gitlab.io/ur_rtde/_static/gripper_2f85.py
Adjusted for use with asyncio

this is to control ROBOTIQ TWO PARALLEL GRIPPER

"""


import asyncio
from enum import Enum
from typing import Union, Tuple, OrderedDict

# TODO: add blocking to release, gripping

class VacuumGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """

    # WRITE VARIABLES (CAN ALSO READ)
    ACT = "ACT"  # Activate the gripper
    GTO = "GTO"  # Go to position
    POS = "POS"  # Target position (0-255, 0 = fully open, 255 = fully closed)
    FOR = "FOR"  # Grip force (0-255)
    SPE = "SPE"  # Grip speed (0-255)
    
    # READ VARIABLES
    STA = "STA"  # Status (0 = reset, 1 = activating, 3 = active)
    OBJ = "OBJ"  # Object detection (0 = moving, 1 = object detected, 2 = object grasped, 3 = no object detected)
    FLT = "FLT"  # Fault (0 = OK, see manual for error codes)

    ENCODING = "UTF-8"  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""

        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""

        MOVING = 0
        DETECTED_MIN = 1
        DETECTED_MAX = 2
        NO_OBJ_DETECTED = 3

    def __init__(self, hostname: str, port: int = 63352) -> None:
        """Constructor.

        :param hostname: Hostname or ip of the robot arm.
        :param port: Port.

        """
        self.socket_reader = None
        self.socket_writer = None
        self.command_lock = asyncio.Lock()

        self.hostname = hostname
        self.port = port

    async def connect(self) -> None:
        """Connects to a gripper on the provided address"""
        self.socket_reader, self.socket_writer = await asyncio.open_connection(self.hostname, self.port)

    async def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket_writer.close()
        await self.socket_writer.wait_closed()
    
    async def _set_var(self, variable: str, value: Union[int, float]) -> bool:
        cmd = f"SET {variable} {value}\n"
        async with self.command_lock:
            self.socket_writer.write(cmd.encode(self.ENCODING))
            await self.socket_writer.drain()
            response = await self.socket_reader.read(1024)
        return self._is_ack(response)
    
    async def _set_var(self, variable: str, value: Union[int, float]) -> bool:
        cmd = f"SET {variable} {value}\n"
        async with self.command_lock:
            self.socket_writer.write(cmd.encode(self.ENCODING))
            await self.socket_writer.drain()
            response = await self.socket_reader.read(1024)
        return self._is_ack(response)

    async def _get_var(self, variable: str) -> int:
        async with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket_writer.write(cmd.encode(self.ENCODING))
            await self.socket_writer.drain()
            data = await self.socket_reader.read(1024)
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data.decode(self.ENCODING)}: does not match '{variable}'")
        return int(value_str)

    @staticmethod
    def _is_ack(data: str) -> bool:
        return data == b"ack"

    async def activate(self) -> None:
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.

        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.
        """
        # await self._set_var(self.ACT, 0)
        await self._set_var(self.ACT, 1)
        while not await self.is_active(): #if self.is_active is True, it willnot go into the loop.
            await asyncio.sleep(0.01)

    async def is_active(self) -> bool:
        """Returns whether the gripper is active."""
        status = await self._get_var(self.STA)
        return VacuumGripper.GripperStatus(status) == VacuumGripper.GripperStatus.ACTIVE
    
    # to close the gripper with low speed 
    async def close_gripper(self, force: int = 100, speed: int = 100) -> None:
        await self._set_var(self.FOR, force)
        await self._set_var(self.SPE, speed)
        await self._set_var(self.POS, 255)
        await self._set_var(self.GTO, 1)

    # to open the gripper with low speed
    async def open_gripper(self, force: int = 100, speed: int = 100) -> None:
        await self._set_var(self.FOR, force)
        await self._set_var(self.SPE, speed)
        await self._set_var(self.POS, 0)
        await self._set_var(self.GTO, 1)

    async def get_object_status(self) -> ObjectStatus:
        a = await self._get_var(self.OBJ)
        return VacuumGripper.ObjectStatus(a)

    async def get_fault_status(self) -> int:
        # value = await self._get_var(self.FLT)
        return await self._get_var(self.FLT)