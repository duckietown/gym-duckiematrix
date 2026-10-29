"""Utilities."""

import numpy as np


def quaternion_to_euler_angles(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Convert a quaternion to Euler angles.

    Args:
        quaternion (tuple[float, float, float, float]): The input
        quaternion.

    Returns:
        tuple[float, float, float]: The corresponding Euler angles
        (roll, pitch, yaw).

    """
    w, x, y, z = quaternion
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Use 90 degrees if out of range
    pitch = np.pi / 2 * np.sign(sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
