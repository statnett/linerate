from functools import partial
from typing import Callable, Optional

import numpy as np

from .units import Ampere, Celsius, FloatOrFloatArray, WattPerMeter

__all__ = ["bisect", "compute_conductor_temperature", "compute_conductor_ampacity"]


def bisect(
    f: Callable[[FloatOrFloatArray], FloatOrFloatArray],
    xmin: FloatOrFloatArray,
    xmax: FloatOrFloatArray,
    tolerance: float,
    invalid_value: Optional[float] = None,
) -> FloatOrFloatArray:
    r"""Compute the roots of a function using a vectorized bisection method.

    Parameters
    ----------
    f:
        :math:`f: \mathbb{R}^n \to \mathbb{R}^n`. Function whose roots we wish to find.
    xmin:
        :math:`x_\min`. Minimum value for the free parameter, :math:`\mathbf{x}`. It is required
        that :math:`\text{sign}(f_i(x_\min)) \neq \text{sign}(f_i(x_\max))`. for all :math:`i`.
    xmax:
        :math:`x_\max`. Maximum value for the free parameter, :math:`\mathbf{x}`. It is required
        that :math:`\text{sign}(f_i(x_\min)) \neq \text{sign}(f_i(x_\max))`. for all :math:`i`.
    tolerance:
        :math:`\Delta x`. The bisection iterations will terminate once all :math:`x_i`-s are
        bounded within an interval of size :math:`\Delta x` or less. The bisection method will
        run for :math:`\left\lceil\frac{x_\max - x_\min}{\Delta x}\right\rceil`
        iterations.
    invalid_value:
        If provided, then the this value is used whenever
        :math:`\text{sign}(f(\mathbf{x}_\min)) = \text{sign}(f(\mathbf{x}_\max))`.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\tilde{\mathbf{x}}`. Estimate for the roots of :math:`f`. For each :math:`f_i`,
        there is a root :math:`x_i \in [\tilde{x}_i - 0.5 \Delta x, \tilde{x}_i + 0.5 \Delta x]`
        so :math:`f_i(x_i) = 0`.
    """
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
    r"""Use the bisection method to compute the steady state conductor temperature.

    Parameters
    ----------
    heat_balance:
        :math:`f(T, A) = P_J + P_s - P_c - P_r~\left[\text{W}~\text{m}^{-1}\right]`. A function of
        both temperature and current that returns the heat balance for the given
        temperature-current pair.
    current:
        :math:`I_\text{max}~\left[\text{A}\right]`. The current flowing through the conductor.
    min_temperature:
        :math:`T_\text{min}~\left[^\circ\text{C}\right]`. Lower bound for the numerical scheme for
        computing the temperature
    max_temperature:
        :math:`T_\text{max}~\left[^\circ\text{C}\right]`. Upper bound for the numerical scheme for
        computing the temperature
    tolerance:
        :math:`\Delta T~\left[^\circ\text{C}\right]`. The numerical accuracy of the
        temperature. The bisection iterations will stop once the numerical temperature
        uncertainty is below :math:`\Delta T`. The bisection method will run for
        :math:`\left\lceil\frac{T_\text{min} - T_\text{min}}{\Delta T}\right\rceil` iterations.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`I~\left[\text{A}\right]`. The thermal rating.
    """
    f = partial(heat_balance, current=current)

    return bisect(f, min_temperature, max_temperature, tolerance)


def compute_conductor_ampacity(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
    max_conductor_temperature: Celsius,
    min_ampacity: Ampere = 0,
    max_ampacity: Ampere = 5_000,
    tolerance: float = 1,  # Ampere
) -> Ampere:
    r"""Use the bisection method to compute the steady-state thermal rating (ampacity).

    Parameters
    ----------
    heat_balance:
        :math:`f(T, A) = P_J + P_s - P_c - P_r~\left[\text{W}~\text{m}^{-1}\right]`. A function of
        both temperature and current that returns the heat balance for the given
        temperature-current pair.
    max_conductor_temperature:
        :math:`T_\text{max}~\left[^\circ\text{C}\right]`. Maximum allowed conductor temperature
    min_ampacity:
        :math:`I_\text{min}~\left[\text{A}\right]`. Lower bound for the numerical scheme for
        computing the ampacity
    max_ampacity:
        :math:`I_\text{min}~\left[\text{A}\right]`. Upper bound for the numerical scheme for
        computing the ampacity
    tolerance:
        :math:`\Delta I~\left[\text{A}\right]`. The numerical accuracy of the ampacity. The
        bisection iterations will stop once the numerical ampacity uncertainty is below
        :math:`\Delta I`. The bisection method will run for
        :math:`\left\lceil\frac{I_\text{min} - I_\text{min}}{\Delta I}\right\rceil` iterations.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`I~\left[\text{A}\right]`. The thermal rating.
    """
    f = partial(heat_balance, max_conductor_temperature)

    return bisect(f, min_ampacity, max_ampacity, tolerance, invalid_value=0)
