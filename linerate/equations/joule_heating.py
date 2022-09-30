from typing import Tuple

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
    T1, T2 = temperature_1, temperature_2
    R1, R2 = resistance_1, resistance_2

    if np.all(R1 == R2):
        return np.ones_like(R1) * R1, np.zeros_like(R1)

    temp = (T1 - 20) * R2 - (T2 - 20) * R1  # CelsiusOhmPerMeter
    rho_20 = np.where(R1 == R2, np.ones_like(R1) * R1, temp / (T1 - T2))
    alpha_20 = np.where(R1 == R2, np.zeros_like(R1), (R1 - R2) / temp)
    return rho_20, alpha_20


def compute_resistance(
    temperature: Celsius,
    resistance_at_20c: OhmPerMeter,
    linear_resistance_coefficient_20c: PerCelsius,
    quadratic_resistance_coefficient_20c: PerSquareCelsius,
) -> OhmPerMeter:
    """Equation (5) on page 15."""
    T_av = temperature
    rho_20 = resistance_at_20c
    alpha_20 = linear_resistance_coefficient_20c
    zeta_20 = quadratic_resistance_coefficient_20c

    return rho_20 * (1 + alpha_20 * (T_av - 20) + zeta_20 * (T_av - 20) ** 2)


def correct_resistance_acsr_magnetic_core_loss(
    ac_resistance: OhmPerMeter,
    current: Ampere,
    aluminium_surface_area: SquareMeter,
    constant_magnetic_effect: Unitless,
    current_density_proportional_magnetic_effect: SquareMeterPerAmpere,
    max_relative_increase: Unitless,
) -> OhmPerMeter:
    """Correct the resistance for AC due to the steel core.

    According to "AC RESISTANCE OF ACSR - MAGNETIC AND TEMPERATUIRE EFFECTS", we can assume a
    linear relationship between the current density and the relative increase in resistance due to
    the steel core of three-layer ACSR. However, they also note that we can assume that the
    increase saturates at 6% (i.e. ``max_relative_increase=1.06``). For two-layer ACSR, this effect
    is negigible and for mono-layer ACSR, the effect behaves differently. Still, some software
    providers use the same correction scheme to correct the magnetic core loss in mono-layer ACSR,
    albeit with a higher saturation point (e.g. 20%).
    """
    I = current  # noqa
    R = ac_resistance
    A = aluminium_surface_area
    b = constant_magnetic_effect
    m = current_density_proportional_magnetic_effect

    if m is None or np.all(m == 0):
        return b * R

    J = I / A
    return np.minimum(b + J * m, max_relative_increase) * R


def compute_joule_heating(current: Ampere, resistance: OhmPerMeter) -> WattPerMeter:
    """Equation (4) on page 14, assuming that the skin effect is incorporated into R."""
    I = current  # noqa
    R = resistance
    return I * I * R
