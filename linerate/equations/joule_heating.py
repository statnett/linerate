from typing import Union

import numpy as np

from ..units import (
    Ampere,
    Celsius,
    OhmPerMeter,
    SquareMeter,
    SquareMeterPerAmpere,
    Unitless,
    WattPerMeter,
)


def compute_resistance(
    conductor_temperature: Celsius,
    temperature1: Celsius,
    temperature2: Celsius,
    resistance_at_temperature1: OhmPerMeter,
    resistance_at_temperature2: OhmPerMeter,
) -> OhmPerMeter:
    r"""Compute the (possibly AC-)resistance of the conductor at a given temperature.

    The resistance is linearly interpolated/extrapolated based on the two temperature-resistance
    measurement pairs provided as arguments.

    Parameters
    ----------
    conductor_temperature:
        :math:`T~\left[^\circ\text{C}\right]`. The average conductor temperature.
    temperature1:
        :math:`T_1~\left[^\circ\text{C}\right]`. The first temperature measurement.
    temperature2:
        :math:`T_2~\left[^\circ\text{C}\right]`. The second temperature measurement.
    resistance_at_temperature1:
        :math:`R_1~\left[\Omega~\text{m}^{-1}\right]`. The resistance at temperature :math:`T=T_1`.
    resistance_at_temperature2:
        :math:`R_2~\left[\Omega~\text{m}^{-1}\right]`. The resistance at temperature :math:`T=T_2`.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`R~\left[\Omega\right]`. The resistance at the given temperature.
    """
    T_av = conductor_temperature
    T_1, T_2 = temperature1, temperature2
    R_1, R_2 = resistance_at_temperature1, resistance_at_temperature2

    a = (R_2 - R_1) / (T_2 - T_1)
    b = R_1 - a * T_1
    return a * T_av + b


def correct_resistance_acsr_magnetic_core_loss(
    ac_resistance: OhmPerMeter,
    current: Ampere,
    aluminium_cross_section_area: SquareMeter,
    constant_magnetic_effect: Union[Unitless, None],
    current_density_proportional_magnetic_effect: Union[SquareMeterPerAmpere, None],
    max_relative_increase: Unitless,
) -> OhmPerMeter:
    r"""Correct for extra resistance in ACSR conductors due to magnetic effects in the steel core.

    Aluminium-conductor steel-reinforced (ACSR) conductors have an additional resistance due to
    magnetic effects in the steel core. Particularly conductors with an odd number of layers with
    aluminium wires.

    According to :cite:p:`ieee.acsr.taskforce`, we can assume a linear relationship between the
    current density and the relative increase in resistance due to the steel core of three-layer
    ACSR. However, the task force also says that we can assume that the increase saturates at
    6%. In :cite:p:`cigre345`, it is stated that the maximum increase for three-layer ACSR is 5%.

    For ACSR with an even number of layers, the effect is negligible since we get cancelling
    magnetic fields, and for mono-layer ACSR, the effect behaves differently. Still, some software
    providers use the same correction scheme for mono-layer ACSR, but with a higher saturation
    point (typically 20%, since that is the maximum resistance increase in mono-layer ACSR
    :cite:p:`cigre345`).

    The linear but saturating increase in resistance leads to the following correction scheme

    .. math::

        R_\text{corrected} = R \min(c_\text{max}, b + mJ),

    where :math:`R_\text{corrected}` is the corrected AC resistance for ACSR conductors,
    :math:`R` is the uncorrected value for the AC resistance at the given temperature,
    :math:`c_\text{max}` is the maximum relative increase (e.g. 6% or :math:`1.06`` for
    three-layer ACSR conductors), :math:`J` is the current density (current divided by the
    aluminium cross-section area, :math:`I/A`) and :math:`b` and :math:`m` are the constant and
    current-density proprtional magnetic effects, respectively (for example obtained from linear
    regression curve).

    Parameters
    ----------
    ac_resistance:
        :math:`R~\left[\Omega\right]`. The AC resistance of the conductor.
    current:
        :math:`I~\left[\text{A}\right]`. The current going through the conductor.
    aluminium_cross_section_area:
        :math:`A_{\text{Al}}~\left[\text{m}^2\right]`. The cross sectional area of the aluminium
        strands in the conductor.
    constant_magnetic_effect:
        :math:`b`. The constant magnetic effect, most likely equal to 1. If ``None``, then no
        correction is used (useful for non-ACSR cables).
    current_density_proportional_magnetic_effect:
        :math:`m`. The current density proportional magnetic effect. If ``None``, then it is
        assumed equal to 0.
    max_relative_increase:
        :math:`c_\text{max}`. Saturation point of the relative increase in conductor resistance.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`R_\text{corrected}~\left[\Omega\right]`. The resistance of the conductor after
        taking steel core magnetization effects into account.
    """
    I = current  # noqa
    R = ac_resistance
    A = aluminium_cross_section_area
    b = constant_magnetic_effect
    m = current_density_proportional_magnetic_effect

    if b is None:
        return R

    if m is None or np.all(m == 0):
        return b * R

    J = I / A
    return np.minimum(b + J * m, max_relative_increase) * R


def compute_joule_heating(current: Ampere, resistance: OhmPerMeter) -> WattPerMeter:
    r"""Compute the Joule heating, assuming AC-resistance for AC lines.

    Parameters
    ----------
    current:
        :math:`I~\left[\text{A}\right]`. The current going through the conductor.
    Resistance:
        :math:`R~\left[\Omega\right]`. The (possibly AC-)resistance of the conductor, correcting
        for all possible magnetisation effects.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`P_J~\left[\text{W}~\text{m}^{-1}\right]`. The Joule heating of the conductor
    """
    I = current  # noqa
    R = resistance
    return I * I * R
