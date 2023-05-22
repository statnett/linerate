import numpy as np

from ..units import Radian, Unitless


def switch_cos_sin(cos_or_sin: Unitless) -> Unitless:
    return np.sqrt(1 - cos_or_sin**2)


def compute_angle_of_attack(angle_1: Radian, angle_2: Radian) -> Radian:
    # TODO: Double check the mathematics here
    angle_1 = (angle_1 % (2 * np.pi)) % np.pi  # TODO: The first modulo may be unnecessary?
    angle_2 = (angle_2 % (2 * np.pi)) % np.pi  # TODO: The first modulo may be unnecessary?
    angle_diff = np.abs(angle_1 - angle_2)
    return np.minimum(angle_diff, np.pi - angle_diff)
