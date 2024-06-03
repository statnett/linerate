import warnings

import numpy as np
from numba import vectorize

from ...units import (
    Celsius,
    KilogramPerCubeMeter,
    Meter,
    MeterPerSecond,
    Radian,
    SquareMeterPerSecond,
    Unitless,
    WattPerMeterPerKelvin,
)

# Physical quantities
#####################


def compute_thermal_conductivity_of_air(film_temperature: Celsius) -> WattPerMeterPerKelvin:
    r"""Approximation of the thermal conductivity of air.

    On page 5 of :cite:p:`cigre207`.

    Parameters
    ----------
    film_temperature:
        :math:`T_f = 0.5 (T_s + T_a)~\left[^\circ\text{C}\right]`. The temperature of the
        thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor sufrace temperature.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\lambda_f~\left[\text{W}~\text{m}^{-1}~\text{K}^{-1}\right]`. The thermal
        conductivity of air at the given temperature.
    """
    T_f = film_temperature
    return 2.42e-2 + 7.2e-5 * T_f


def compute_relative_air_density(height_above_sea_level: Meter) -> Unitless:
    r"""Approximation of the relative density of air at a given altitude,
    relative to density at sea level.

    Equation on page 6 of :cite:p:`cigre207`.

    Parameters
    ----------
    height_above_sea_level:
        :math:`y~\left[\text{m}\right]`. The conductor's altitude.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\rho_r`. The relative mass density of air.
    """
    y = height_above_sea_level
    return np.exp(-1.16e-4 * y)


def compute_kinematic_viscosity_of_air(film_temperature: Celsius) -> KilogramPerCubeMeter:
    r"""Approximation of the kinematic viscosity of air at a given temperature.

    Equation on page 5 of :cite:p:`cigre207`.

    Parameters
    ----------
    film_temperature:
        :math:`T_f = 0.5 (T_s + T_a)~\left[^\circ\text{C}\right]`. The temperature of the
        thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor sufrace temperature.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.
    """
    T_f = film_temperature
    return 1.32e-5 + 9.5e-8 * T_f


def compute_prandtl_number(
    film_temperature: Celsius,
) -> Unitless:
    r"""Compute the Prandtl number.

    Defined on page 5 of :cite:p:`cigre207`.

    The Prandtl number measures the ratio between viscosity and thermal diffusivity for a fluid.

    Parameters
    ----------
    film_temperature:
        :math:`T_f = 0.5 (T_s + T_a)~\left[^\circ\text{C}\right]`. The temperature of the
        thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor sufrace temperature.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Pr}`. The Prandtl number.
    """
    return 0.715 - 2.5e-4 * film_temperature


def compute_reynolds_number(
    wind_speed: MeterPerSecond,
    conductor_diameter: Meter,
    kinematic_viscosity_of_air: SquareMeterPerSecond,
    relative_air_density: Unitless,
) -> Unitless:
    r"""Compute the Reynolds number using the conductor diameter as characteristic length scale.

    Defined on page 5 of :cite:p:`cigre207`.
    This is a non-standard definition which seems to indicate that the kinematic viscosity has to
    be corrected for the density.

    Parameters
    ----------
    wind_speed:
        :math:`v~\left[\text{m}~\text{s}^{-1}\right]`. The wind speed.
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    kinematic_viscosity_of_air:
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.
    relative_air_density:
        :math:`\rho_r~1`. The air density relative to density at sea level.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Re}`. The Reynolds number.
    """
    v = wind_speed
    D = conductor_diameter
    nu_f = kinematic_viscosity_of_air
    rho_r = relative_air_density
    return rho_r * v * D / nu_f


## Nusselt number calculation
#############################


def _check_perpendicular_flow_nusseltnumber_out_of_bounds(reynolds_number):
    Re = reynolds_number
    if np.any(Re < 0):
        raise ValueError("Reynolds number cannot be negative.")

    if np.any(np.logical_or(Re < 100, Re > 5e4)):
        warnings.warn("Reynolds number is out of bounds", stacklevel=5)


@vectorize(nopython=True)
def _compute_perpendicular_flow_nusseltnumber(
    reynolds_number: Unitless,
    conductor_roughness: Meter,
) -> Unitless:
    # From table on page 6 in Cigre207
    Re = reynolds_number
    Rs = conductor_roughness

    if Re < 100:
        B, n = 0, 0
    elif Re < 2.65e3:
        B, n = 0.641, 0.471
    elif Rs <= 0.05:
        B, n = 0.178, 0.633
    else:
        B, n = 0.048, 0.800

    return B * Re**n  # type: ignore


def compute_perpendicular_flow_nusseltnumber(
    reynolds_number: Unitless,
    conductor_roughness: Meter,
) -> Unitless:
    r"""Compute the Nusselt number for perpendicular flow.

    The Nusselt number is the ratio of conductive heat transfer to convective heat transfer.

    Parameters
    ----------
    reynolds_number:
        :math:`\text{Re}`. The Reynolds number.
    conductor_roughness:
        :math:`\text{Rs}`. The roughness number

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Nu}_{90}`. The perpendicular flow Nusselt number.
    """
    _check_perpendicular_flow_nusseltnumber_out_of_bounds(reynolds_number)
    return _compute_perpendicular_flow_nusseltnumber(
        reynolds_number,
        conductor_roughness,
    )


def compute_low_wind_speed_nusseltnumber(
    perpendicular_flow_nusselt_number: Unitless,
) -> Unitless:
    r"""Compute the corrected Nusselt number for low wind speed.

    Parameters
    ----------
    perpendicular_flow_nusselt_number:
        :math:`\text{Nu}_{90}`. The perpendicular flow Nusselt number.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Nu}_{90}`. The corrected Nusselt number for low wind speed.
    """
    return 0.55 * perpendicular_flow_nusselt_number


@vectorize(nopython=True)
def _correct_wind_direction_effect_on_nusselt_number(
    perpendicular_flow_nusselt_number: Unitless,
    angle_of_attack: Radian,
) -> Unitless:
    delta = angle_of_attack
    Nu_90 = perpendicular_flow_nusselt_number

    sin_delta = np.sin(delta)
    # Equation (14) on page 7 of Cigre207
    if delta <= np.radians(24):
        correction_factor = 0.42 + 0.68 * (sin_delta**1.08)
    else:
        correction_factor = 0.42 + 0.58 * (sin_delta**0.90)

    return correction_factor * Nu_90


def correct_wind_direction_effect_on_nusselt_number(
    perpendicular_flow_nusselt_number: Unitless,
    angle_of_attack: Radian,
) -> Unitless:
    r"""Correct the Nusselt number for the wind's angle-of-attack.

    Equation (14) on page 7 of :cite:p:`cigre207`.

    The perpendicular flow nusselt number is denoted as :math:`\text{Nu}_\delta` in
    :cite:p:`cigre207` since the wind's angle of attack is :math:`\delta`.

    Parameters
    ----------
    perpendicular_flow_nusselt_number:
        :math:`\text{Nu}_{90}`. The perpendicular flow Nusselt number.
    angle_of_attack:
        :math:`\delta~\left[\text{radian}\right]`. The wind angle-of-attack.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Nu}_\delta`. The Nusselt number for the given wind angle-of-attack.
    """
    return _correct_wind_direction_effect_on_nusselt_number(
        perpendicular_flow_nusselt_number, angle_of_attack
    )


## Natural convection computations (no wind):
#############################################


def _check_horizontal_natural_nusselt_number(
    grashof_number: Unitless, prandtl_number: Unitless
) -> None:
    GrPr = grashof_number * prandtl_number
    if np.any(GrPr < 0):
        raise ValueError("GrPr cannot be negative.")
    elif np.any(GrPr > 1e6):
        raise ValueError("GrPr out of bounds: Must be < 10^6.")


@vectorize(nopython=True)
def _compute_horizontal_natural_nusselt_number(
    grashof_number: Unitless,
    prandtl_number: Unitless,
) -> Unitless:
    GrPr = grashof_number * prandtl_number

    if GrPr < 1e2:
        # Outside table range, should we use 0??
        return 0
    elif GrPr < 1e4:
        return 0.850 * GrPr**0.188
    elif GrPr < 1e6:
        return 0.480 * GrPr**0.250
    else:
        # Outside table range, what should we do here?
        return 0.125 * GrPr**0.333


def compute_horizontal_natural_nusselt_number(
    grashof_number: Unitless,
    prandtl_number: Unitless,
) -> Unitless:
    r"""The Nusselt number for natural (passive) convection on a horizontal conductor.

    Equation (16) and Table II on page 7 of :cite:p:`cigre207`.

    The coefficient table is modified slightly so coefficients with
    :math:`\text{Gr}\text{Pr} < 0.1` leads to :math:`\text{Nu} = 0`.

    Parameters
    ----------
    grashof_number:
        :math:`\text{Gr}`. The Grashof number.
    prandtl_number:
        :math:`\text{Pr}`. The Prandtl number.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Nu}_0`. The natural convection nusselt number assuming horizontal conductor.
    """
    _check_horizontal_natural_nusselt_number(grashof_number, prandtl_number)
    return _compute_horizontal_natural_nusselt_number(
        grashof_number,
        prandtl_number,
    )


def compute_nusselt_number(
    forced_convection_nusselt_number: Unitless,
    natural_nusselt_number: Unitless,
    low_wind_nusselt_number: Unitless,
    wind_speed: MeterPerSecond,
) -> Unitless:
    r"""Compute the nusselt number.

    Described on page 7 of :cite:p:`cigre207`.

    Parameters
    ----------
    forced_convection_nusselt_number:
        :math:`\text{Nu}_\delta`. The Nusselt number for the given wind angle-of-attack.
    natural_nusselt_number:
        :math:`\text{Nu}_\delta`. The natural convection nusselt number for horizontal conductor.
    low_wind_nusselt_number:
        :math:`\text{Nu}_cor`. Corrected Nusselt number for low wind.
    wind_speed:
        :math:`v~\left[\text{m}~\text{s}^{-1}\right]`. The wind speed.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`Nu`. The nusselt number.
    """
    normal_nusselt = np.maximum(forced_convection_nusselt_number, natural_nusselt_number)
    low_wind_nusselt = np.maximum(normal_nusselt, low_wind_nusselt_number)
    low_wind_speed = wind_speed < 0.5
    return np.where(low_wind_speed, low_wind_nusselt, normal_nusselt)
