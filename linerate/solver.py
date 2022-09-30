from functools import partial
from typing import Callable, Optional

import numpy as np

from .units import Ampere, Celsius, FloatOrFloatArray, WattPerMeter

__all__ = ["compute_conductor_temperature", "compute_conductor_ampacity"]


def bisect(
    f: Callable[[FloatOrFloatArray], FloatOrFloatArray],
    xmin: FloatOrFloatArray,
    xmax: FloatOrFloatArray,
    tolerance: float,
    invalid_value: Optional[float] = None,
) -> FloatOrFloatArray:
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("xmin and xmax must be finite.")
    interval = np.max(np.abs(xmax - xmin))

    f_left = f(xmin)
    f_right = f(xmax)

    invalid_mask = np.sign(f_left) == np.sign(f_right)
    if np.any(invalid_mask) and invalid_value is None:
        raise ValueError(
            "f(xmin) and f(xmax) have the same sign. Consider increasing the search interval."
        )
    elif isinstance(invalid_mask, bool) and invalid_mask:
        return invalid_value  # type: ignore

    while interval > tolerance:
        xmid = 0.5 * (xmax + xmin)
        interval *= 0.5
        f_mid = f(xmid)

        mask = f_mid * f_left > 0  # fast way to check sign(f_mid) == sign(f_left)
        xmin = np.where(mask, xmid, xmin)
        xmax = np.where(mask, xmax, xmid)
        f_left = np.where(mask, f_mid, f_left)
        f_right = np.where(mask, f_right, f_mid)

    out = np.where(invalid_mask, invalid_value, 0.5 * (xmax + xmin))  # type: ignore
    return out


def compute_conductor_temperature(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
    current: Ampere,
    min_temperature: Celsius = -30,
    max_temperature: Celsius = 150,
    tolerance: float = 0.5,  # Celsius
) -> Celsius:
    f = partial(heat_balance, current=current)

    return bisect(f, min_temperature, max_temperature, tolerance)


def compute_conductor_ampacity(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
    max_conductor_temperature: Celsius,
    min_ampacity: Ampere = 0,
    max_ampacity: Ampere = 5_000,
    tolerance: float = 1,  # Ampere
) -> Ampere:
    f = partial(heat_balance, max_conductor_temperature)

    return bisect(f, min_ampacity, max_ampacity, tolerance, invalid_value=0)
