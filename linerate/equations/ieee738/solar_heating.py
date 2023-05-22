import numpy as np

from linerate.equations.math import switch_cos_sin
from linerate.units import (
    BoolOrBoolArray,
    Meter,
    Radian,
    Unitless,
    WattPerMeter,
    WattPerSquareMeter,
)

_clear_atmosphere_polynomial = np.poly1d(
    [-4.07608e-9, 1.94318e-6, -3.61118e-4, 3.46921e-2, -1.9220, 63.8044, -42.2391]
)


_industrial_atmosphere_polynomial = np.poly1d(
    [1.3236e-8, -4.3446e-6, 5.4654e-4, -3.1658e-2, 6.6138e-1, 14.2110, 53.1821]
)


def compute_total_heat_flux_density(
    sin_solar_altitude: Radian,
    clear_atmosphere: BoolOrBoolArray,
) -> WattPerSquareMeter:
    r"""Compute the heat flux density received by a surface at sea level.

    Equation (18) on page 19 of :cite:p:`ieee738`.

    This function takes in the sin of the solar altitude, :math`H_c`, in radians.
    This is because this is what is calculated in compute_sin_solar_altitude.
    This function therefore takes the arcsin of sin_solar_altitude, and then
    converts it do degrees.

    Parameters
    ----------
    sin_solar_altitude:
        :math:`sin(H_c)~\left[\text{radian}~\right]`. The sin of the solar altitude.
    clear_atmosphere:
        True or False. True: clear atmosphere. False: industrial atmosphere.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`Q_s~\left[\text{W}~\text{m}^{-2}\right]`
    """
    sin_H_c = sin_solar_altitude
    H_c = np.degrees(np.arcsin(sin_H_c))
    Q_S = np.where(
        clear_atmosphere,
        _clear_atmosphere_polynomial(H_c),
        _industrial_atmosphere_polynomial(H_c),
    )
    return np.where(Q_S >= 0, Q_S, 0 * Q_S)


def compute_solar_altitude_correction_factor(
    height_above_sea_level_of_conductor: Meter,
) -> Unitless:
    r"""Compute the solar altitude correction factor.

    Equation (20) on page 20 of :cite:p:`ieee738`.

    Parameters
    ----------
    height_above_sea_level_of_conductor:
        :math:`H_e~\left[\text{m}~\right]`. The elevation of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`K_{solar}`
    """
    H_e = height_above_sea_level_of_conductor
    A = 1
    B = 1.148e-4
    C = -1.108e-8
    return np.poly1d([C, B, A])(H_e)


def compute_elevation_correction_factor(
    solar_altitude_correction_factor: Unitless,
    total_heat_flux_density: WattPerSquareMeter,
) -> WattPerSquareMeter:
    r"""Compute the elevation correction factor.

    Equation (19) on page 19 of :cite:p:`ieee738`.

    The equation is used to correct the solar heat intensity for altitude.

    Parameters
    ----------
    solar_altitude_correction_factor:
        :math:`K_{solar}\left[ \right]`
    total_heat_flux_density:
        :math:`Q_s~\left[\text{W}~\text{m}^{-2}\right]`

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`Q_{se}~\left[\text{W}~\text{m}^{-2}\right]`. The elevation correction factor.
    """
    K_solar = solar_altitude_correction_factor
    Q_s = total_heat_flux_density
    return K_solar * Q_s


def compute_solar_heating(
    absorptivity: Unitless,
    elevation_correction_factor: WattPerSquareMeter,
    cos_solar_effective_incidence_angle: Radian,
    projected_area_of_conductor: Meter,  # Meter squared per linear meter
) -> WattPerMeter:
    r"""Compute the solar heating experienced by the ocnductor.

    Equation (8) on page 9 of :cite:p:`ieee738`.

    Parameters
    ----------
    absorptivity:
        :math:`\alpha`. Material constant. According to :cite:p:`ieee738`, it has a range from
        0.23 to 0.91, with new conductors having a value between 0.2 and 0.4, and over time
        increasing to between 0.5 and 0.9.
    elevation_correction_factor:
        :math:`Q_{se}~\left[\text{W}~\text{m}^{-2}\right]`.The elevation correction factor.
    cos_solar_effective_incidence_angle:
        :math:`cos(\theta)~\left[\text{radian}\right]`. The cosine of the effective angle of
        incidence of the sun’s rays.
    projected_area_of_conductor:
        :math:`A~\left[\text{m}\right]`. :math:`\text{m}^2` per linear m. Equal to the outer
        diameter of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`P_S~\left[\text{W}~\text{m}^{-1}\right]`. The solar heating of the conductor.
    """
    alpha = absorptivity
    Q_se = elevation_correction_factor
    sin_theta = switch_cos_sin(cos_solar_effective_incidence_angle)
    A = projected_area_of_conductor
    return alpha * Q_se * sin_theta * A
