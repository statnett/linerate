import numpy as np

from ..units import Radian, Unitless


def switch_cos_sin(cos_or_sin: Unitless) -> Unitless:
    return np.sqrt(1 - cos_or_sin**2)


def compute_angle_of_attack(angle_1: Radian, angle_2: Radian) -> Radian:
    angle_diff = np.abs(angle_1 - angle_2) % np.pi
    return np.minimum(angle_diff, np.pi - angle_diff)
