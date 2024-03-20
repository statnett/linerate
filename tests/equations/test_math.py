import hypothesis
import hypothesis.strategies as st
import numpy as np
from pytest import approx, mark

import linerate.equations.math as cigre_math


@hypothesis.given(angle=st.floats())
def test_switch_cos_sin(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    sin_from_cos = cigre_math.switch_cos_sin(cos)
    cos_from_sin = cigre_math.switch_cos_sin(sin)

    if np.isnan(angle) or np.isinf(angle):
        assert np.isnan(sin_from_cos)
        assert np.isnan(cos_from_sin)
    else:
        assert np.abs(cos) == approx(cos_from_sin, abs=1e-8)
        assert np.abs(sin) == approx(sin_from_cos, abs=1e-8)


@mark.parametrize(
    "angle_1, angle_2, expected",
    [
        (0, 4 * np.pi, 0),  # Handles differences in angle of more than 2 * pi
        (0, 3 * np.pi / 4, np.pi / 4),  # Finds lowest candidate angle (result <= pi/2)
        (-np.pi, -np.pi / 2, np.pi / 2),  # Converts negative angles to positive
    ],
    ids=[
        "Angle between 0 and 4*pi (720 degrees) is equivalent to 0",
        "Angle between 0 and 3*pi/4 (135 degrees) is equivalent to pi/4 (45 degrees)",
        "Angle between -pi (-180 degrees) and -pi/2 (-90 deg) is equivalent to pi/2 (90 deg)",
    ],
)
def test_compute_angle_of_attack(angle_1, angle_2, expected):
    result = cigre_math.compute_angle_of_attack(angle_1, angle_2)
    assert result == expected
