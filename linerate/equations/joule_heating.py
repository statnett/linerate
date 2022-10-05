from typing import Tuple, Union

import numpy as np

from ..units import (
    Ampere,
    Celsius,
    OhmPerMeter,
    PerCelsius,
    PerSquareCelsius,
    SquareMeter,
    SquareMeterPerAmpere,
    Unitless,
    WattPerMeter,
)


def compute_linear_resistance_parameters(
    temperature_1: Celsius,
    resistance_1: OhmPerMeter,
    temperature_2: Celsius,
    resistance_2: OhmPerMeter,
) -> Tuple[OhmPerMeter, PerCelsius]:
    r"""Convert from resistance at two known temperatures to the linear resistance parameters.

    Parameters
    ----------
    temperature_1:
        :math:`T_1~\left[^\circ\text{C}\right]`. The first known temperature
    resistance_1:
        :math:`R_1~\left[^\Omega_1\right]`. The first known resistance
    temperature_2:
        :math:`T_2~\left[^\circ\text{C}\right]`. The second known temperature
    resistance_2:
        :math:`R_2~\left[^\Omega_1\right]`. The second known resistance

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\rho_{20}~\left[\Omega~\text{m}^{-1}\right]`. The resistance at
        :math:`20^\circ \text{C}`.
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\alpha_{20}~\left[\left(^\circ\text{C}\right)^{-1}\right]`.
    """
    T1, T2 = temperature_1, temperature_2
    R1, R2 = resistance_1, resistance_2

    if np.all(R1 == R2):
        return np.ones_like(R1) * R1, np.zeros_like(R1)

    temp = (T1 - 20) * R2 - (T2 - 20) * R1  # CelsiusOhmPerMeter
    rho_20 = np.where(R1 == R2, np.ones_like(R1) * R1, temp / (T1 - T2))
    alpha_20 = np.where(R1 == R2, np.zeros_like(R1), (R1 - R2) / temp)
    return rho_20, alpha_20


def compute_resistance(
    conductor_temperature: Celsius,
    resistance_at_20c: OhmPerMeter,
    linear_resistance_coefficient_20c: PerCelsius,
    quadratic_resistance_coefficient_20c: PerSquareCelsius,
) -> OhmPerMeter:
    r"""Compute the (possibly AC-)resistance of the conductor at a given temperature.
    
    Equation (5) on page 15 of :cite:p:`cigre601` states that the conductor resistance, :math:`R`
    is given by

    .. math::

        R = \rho_{20} \left(
            1 +
            \alpha_{20}\left(T - 20^\circ \text{C}\right) +
            \zeta_{20}\left(T - 20^\circ \text{C}\right)^2
        \right),
    
    where :math:`T` is the conductor temperature and :math:`\rho_{20}, \alpha_{20}` and
    :math:`\zeta_{20}` are parameters for the given conductor.
    
    Parameters
    ----------
    conductor_temperature:
        :math:`T~\left[^\circ\text{C}\right]`. The average conductor temperature.
    resistance_at_20c:
        :math:`\rho_{20}~\left[\Omega~\text{m}^{-1}\right]`. The resistance at
        :math:`20^\circ \text{C}`.
    linear_resistance_coefficient_20c:
        :math:`\alpha_{20}~\left[\left(^\circ\text{C}\right)^{-1}\right]`.
    quadratic_resistance_coefficient_20c:
        :math:`\zeta_{20}~\left[\left(^\circ\text{C}\right)^{-2}\right]`.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`R~\left[\Omega\right]`. The resistance at the given temperature.
    """
    T_av = conductor_temperature
    rho_20 = resistance_at_20c
    alpha_20 = linear_resistance_coefficient_20c
    zeta_20 = quadratic_resistance_coefficient_20c

    return rho_20 * (1 + alpha_20 * (T_av - 20) + zeta_20 * (T_av - 20) ** 2)


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
    6%. For ACSR with an even number of layers, the effect is negligible since we get cancelling
    magnetic fields, and for mono-layer ACSR, the effect behaves differently. Still, some software
    providers use the same correction scheme for mono-layer ACSR, but with a higher saturation
    point (e.g. 20%).

    This leads to the following correction scheme

    .. math::

        R_\text{corrected} = R \min(c_\text{max}, b + mJ),
    
    where :math:`R_\text{corrected}` is the corrected AC resistance for ACSR conductors,
    :math:`R` is the uncorrected value for the AC resistance at the given temperature,
    :math:`c_\text{max}` is the maximum relative increase (typically 6% or :math:`1.06`` for
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
        :math:`A~\left[\text{m}^2\right]`. The cross sectional area of the aluminium strands in
        the conductor.
    constant_magnetic_effect:
        :math:`b`. The constant magnetic effect, most likely equal to 1. If ``None``, then no
        no correction is used (useful for non-ACSR cables).
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
        :math:`P_j~\left[\text{W}~\text{m}^{-1}\right]`. The joule heating of the conductor
    """
    I = current  # noqa
    R = resistance
    return I * I * R
