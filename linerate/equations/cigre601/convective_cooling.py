import warnings
from textwrap import dedent

import numpy as np
from numba import vectorize

from ...units import (
    Celsius,
    JoulePerKilogramPerKelvin,
    KilogramPerCubeMeter,
    KilogramPerMeterPerSecond,
    Meter,
    MeterPerSecond,
    MeterPerSquareSecond,
    Radian,
    SquareMeterPerSecond,
    Unitless,
    WattPerMeter,
    WattPerMeterPerKelvin,
)

# Physical quantities
#####################


def compute_temperature_gradient(
    total_heat_gain: WattPerMeter,
    conductor_thermal_conductivity: WattPerMeterPerKelvin,
    core_diameter: Meter,
    conductor_diameter: Meter,
) -> Celsius:
    r"""Compute the difference between the core and surface temperature.

    Equation (15) & (16) on page 22 of :cite:p:`cigre601`.

    Parameters
    ----------
    total_heat_gain:
        :math:`P_T = I^2 R~\left[\text{W}~\text{m}^{-1}\right]`. The Joule heating of the
        conductor (see p. 81 of :cite:p:`cigre601`).
    conductor_thermal_conductivity:
        :math:`\lambda \left[\text{W}~\text{m}^{-1}~\text{K}^{-1}\right]`. The effective
        conductor thermal conductivity. It is usually between :math:`0.5` and
        :math:`7~W~m^{-1}~K^{-1}`. Recommended values are
        :math:`0.7~\text{W}~\text{m}^{-1}~\text{K}^{-1}` for conductors with no tension on the
        aluminium strands and :math:`1.5~\text{W}~\text{m}^{-1}~\text{K}^{-1}` for conductors
        with aluminium strands under a tension of at least 40 N :cite:p:`cigre601`.
    core_diameter:
        :math:`D_1~\left[\text{m}\right]`. Diameter of the steel core of the conductor.
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`T_c - T_s~\left[^\circ\text{C}\right]`. The difference between the core and
        surface temperature.
    """
    lambda_ = conductor_thermal_conductivity
    D_1 = core_diameter
    D = conductor_diameter
    P_T = total_heat_gain
    pi = np.pi

    tmp = P_T / (2 * pi * lambda_)

    if D_1 == 0:  # TODO: Maybe lower tolerance?
        return 0.5 * tmp
    else:
        D_1_sq = D_1**2
        delta_D_sq = D**2 - D_1_sq
        return tmp * (0.5 - (D_1_sq / delta_D_sq) * np.log(D / D_1))


def compute_thermal_conductivity_of_air(film_temperature: Celsius) -> WattPerMeterPerKelvin:
    r"""Approximation of the thermal conductivity of air up to :math:`300 ^\circ\text{C}`.

    Equation (18) on page 24 of :cite:p:`cigre601`.

    Compared with table values from textbook, which showed a good approximation.

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
    return 2.368e-2 + 7.23e-5 * T_f - 2.763e-8 * (T_f**2)


def compute_air_density(
    film_temperature: Celsius, height_above_sea_level: Meter
) -> KilogramPerCubeMeter:
    r"""Approximation of the density of air at a given temperature and altitude.

    Equation (20) on page 25 of :cite:p:`cigre601`.

    Parameters
    ----------
    film_temperature:
        :math:`T_f = 0.5 (T_s + T_a)~\left[^\circ\text{C}\right]`. The temperature of the
        thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor sufrace temperature.
    height_above_sea_level:
        :math:`y~\left[\text{m}\right]`. The conductor's altitude.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\gamma~\left[\text{kg}~\text{m}^{-3}\right]`. The mass density of air.
    """
    T_f = film_temperature
    y = height_above_sea_level
    return (1.293 - 1.525e-4 * y + 6.379e-9 * (y**2)) / (1 + 0.00367 * T_f)


def compute_dynamic_viscosity_of_air(film_temperature: Celsius) -> KilogramPerMeterPerSecond:
    r"""Approximation of the dynamic viscosity of air at a given temperature.

    Equation (19) on page 25 of :cite:p:`cigre601`.

    Parameters
    ----------
    film_temperature:
        :math:`T_f = 0.5 (T_s + T_a)~\left[^\circ\text{C}\right]`. The temperature of the
        thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor sufrace temperature.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\mu_f~\left[\text{kg}~\text{m}^{-1}~\text{s}^{-1}\right]`. The dynamic viscosity
        of air.
    """
    T_f = film_temperature
    return 17.239e-6 + 4.635e-8 * T_f - 2.03e-11 * (T_f**2)


def compute_kinematic_viscosity_of_air(
    dynamic_viscosity_of_air: KilogramPerMeterPerSecond, air_density: KilogramPerCubeMeter
) -> SquareMeterPerSecond:
    r"""Compute the kinematic viscosity of air.

    Definition in text on page 25 of :cite:p:`cigre601`.

    Parameters
    ----------
    dynamic_viscosity_of_air:
        :math:`\mu_f~\left[\text{kg}~\text{m}^{-1}~\text{s}^{-1}\right]`. The dynamic viscosity of
        air.
    air_density:
        :math:`\gamma~\left[\text{kg}~\text{m}^{-3}\right]`. The mass density of air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.
    """
    return dynamic_viscosity_of_air / air_density


# Unitless quantities
#####################


def compute_reynolds_number(
    wind_speed: MeterPerSecond,
    conductor_diameter: Meter,
    kinematic_viscosity_of_air: SquareMeterPerSecond,
) -> Unitless:
    r"""Compute the Reynolds number using the conductor diameter as characteristic length scale.

    Defined in the text on page 25 of :cite:p:`cigre601`.

    The Reynolds number is a dimensionless quantity that can be used to assess if a stream is
    likely to be turbulent or not. It is given by

    .. math::

        \text{Re} = \frac{v L}{\nu},

    where :math:`v` is the flow velocity, :math:`L` is a *characteristic length* (in our case,
    the conductor diameter) and :math:`\nu` is the kinematic viscosity.

    Parameters
    ----------
    wind_speed:
        :math:`v~\left[\text{m}~\text{s}^{-1}\right]`. The wind speed.
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    kinematic_viscosity_of_air:
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Re}`. The Reynolds number.
    """
    v = wind_speed
    D = conductor_diameter
    nu_f = kinematic_viscosity_of_air
    return v * D / nu_f


def compute_grashof_number(
    conductor_diameter: Meter,
    surface_temperature: Celsius,
    air_temperature: Celsius,
    kinematic_viscosity_of_air: SquareMeterPerSecond,
    coefficient_of_gravity: MeterPerSquareSecond = 9.807,
) -> Unitless:
    r"""Compute the Grashof number.

    Defined in the nomenclature on page 7 of :cite:p:`cigre601`.

    The Grashof number is a dimensionless quantity that can be used to assess the degree of free
    and forced convective heat transfer.

    Parameters
    ----------
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    surface_temperature:
        :math:`T_s~\left[^\circ\text{C}\right]`. The conductor surface temperature.
    air_temperature:
        :math:`T_a~\left[^\circ\text{C}\right]`. The ambient air temperature.
    kinematic_viscosity_of_air:
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.
    coefficient_of_gravity:
        :math:`g~\left[\text{m}~\text{s}^{-2}\right]`. The graviatational constant, optional
        (default=9.807).

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Gr}`. The Grashof number.
    """
    T_s = surface_temperature
    T_a = air_temperature
    T_f = 0.5 * (T_s + T_a)
    D = conductor_diameter
    nu_f = kinematic_viscosity_of_air
    g = coefficient_of_gravity

    return (D**3) * np.abs((T_s - T_a)) * g / ((T_f + 273.15) * (nu_f**2))


def compute_prandtl_number(
    thermal_conductivity_of_air: WattPerMeterPerKelvin,
    dynamic_viscosity_of_air: KilogramPerMeterPerSecond,
    specific_heat_capacity_of_air: JoulePerKilogramPerKelvin,
) -> Unitless:
    r"""Compute the Prandtl number.

    Defined in the nomenclature on page 8 of :cite:p:`cigre601`.

    The Prandtl number measures the ratio between viscosity and thermal diffusivity for a fluid.

    Parameters
    ----------
    thermal_conductivity_of_air:
        :math:`\lambda_f~\left[\text{W}~\text{m}^{-1}~\text{K}^{-1}\right]`. The thermal
        conductivity of air at the given temperature.
    dynamic_viscosity_of_air:
        :math:`\mu_f~\left[\text{kg}~\text{m}^{-1}~\text{s}^{-1}\right]`. The dynamic viscosity of
        air.
    specific_heat_capacity_of_air:
        :math:`\text{J}~\left[\text{kg}^{-1}~\text{K}^{-1}\right]`. The specific heat capacity of
        air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Pr}`. The Prandtl number.
    """
    lambda_f = thermal_conductivity_of_air
    mu_f = dynamic_viscosity_of_air
    c_f = specific_heat_capacity_of_air

    return c_f * mu_f / lambda_f


def compute_conductor_roughness(
    conductor_diameter: Meter,
    outer_layer_strand_diameter: Meter,
) -> Unitless:
    r"""Compute the surface roughness of the conductor.

    Defined in the text on page 25 of :cite:p:`cigre601`.

    Parameters
    ----------
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    outer_layer_strand_diameter:
        :math:`d~\left[\text{m}\right]`. The diameter of the strands in the outer layer of the
        conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Rs}`. The roughness number
    """
    D = conductor_diameter
    d = outer_layer_strand_diameter

    if np.any(d < 0):
        raise ValueError(
            dedent(
                """\
            Cannot have negative outer layer strand diameter. If it was set this way to signify
            that it is a smooth conductor, then set the outer layer strand diameter to nan or zero
            instead.\
            """
            )
        )
    if np.any(d >= D):
        raise ValueError(
            "The outer layer strand diameter must be strictly smaller than the conductor diameter."
        )

    return d / (2 * (D - d))


## Nusselt number calculation
#############################


def _check_perpendicular_flow_nusseltnumber_out_of_bounds(reynolds_number, conductor_roughness):
    Re = reynolds_number
    Rs = conductor_roughness
    if np.any(Re < 0):
        raise ValueError("Reynolds number cannot be negative.")

    smooth_mask = np.logical_or(Rs == 0, np.isnan(Rs))
    smooth_out_of_bounds = Re > 200_000
    rough_out_of_bounds = np.logical_and(np.logical_not(smooth_mask), Re > 50_000)
    out_of_bounds = np.logical_or(smooth_out_of_bounds, rough_out_of_bounds)

    if np.any(out_of_bounds):
        warnings.warn("Reynolds number is out of bounds", stacklevel=5)


@vectorize(nopython=True)
def _compute_perpendicular_flow_nusseltnumber(
    reynolds_number: Unitless,
    conductor_roughness: Meter,
) -> Unitless:
    # TODO: Look at references for this table
    Re = reynolds_number
    Rs = conductor_roughness

    if Rs == 0 or np.isnan(Rs):
        if Re < 35:
            B, n = 0, 0
        elif Re < 5000:
            B, n = 0.583, 0.471
        elif Re < 50_000:
            B, n = 0.148, 0.633
        else:
            B, n = 0.0208, 0.814
    elif Rs <= 0.05:
        if Re < 100:
            B, n = 0, 0
        elif Re < 2650:
            B, n = 0.641, 0.471
        else:
            B, n = 0.178, 0.633
    else:
        if Re < 100:
            B, n = 0, 0
        elif Re < 2650:
            B, n = 0.641, 0.471
        else:
            B, n = 0.048, 0.800

    return B * Re**n  # type: ignore


def compute_perpendicular_flow_nusseltnumber(
    reynolds_number: Unitless,
    conductor_roughness: Meter,
) -> Unitless:
    r"""Compute the Nusselt number for perpendicular flow.

    Equation (21) and Table 4 on pages 25-26 of :cite:p:`cigre601`.

    The perpendicular flow nusselt number is denoted as :math:`\text{Nu}_{90}` in :cite:p:`cigre601`
    since the wind's angle of attack is :math:`90^\circ`.

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
    _check_perpendicular_flow_nusseltnumber_out_of_bounds(reynolds_number, conductor_roughness)
    return _compute_perpendicular_flow_nusseltnumber(
        reynolds_number,
        conductor_roughness,
    )


@vectorize(nopython=True)
def _correct_wind_direction_effect_on_nusselt_number(
    perpendicular_flow_nusselt_number: Unitless,
    angle_of_attack: Radian,
    conductor_roughness: Unitless,
) -> Unitless:
    delta = angle_of_attack
    Nu_90 = perpendicular_flow_nusselt_number
    Rs = conductor_roughness

    sin_delta = np.sin(delta)

    if Rs == 0 or np.isnan(Rs):
        sin_delta_sq = sin_delta**2
        cos_delta_sq = 1 - sin_delta_sq

        correction_factor = (sin_delta_sq + cos_delta_sq * 0.0169) ** 0.225
    else:
        if delta <= np.radians(24):
            correction_factor = 0.42 + 0.68 * (sin_delta**1.08)
        else:
            correction_factor = 0.42 + 0.58 * (sin_delta**0.90)

    return correction_factor * Nu_90


def correct_wind_direction_effect_on_nusselt_number(
    perpendicular_flow_nusselt_number: Unitless,
    angle_of_attack: Radian,
    conductor_roughness: Unitless,
) -> Unitless:
    r"""Correct the Nusselt number for the wind's angle-of-attack.

    Equation (21) and Table 4 on pages 25-26 of :cite:p:`cigre601`.

    The perpendicular flow nusselt number is denoted as :math:`\text{Nu}_\delta` in
    :cite:p:`cigre601` since the wind's angle of attack is :math:`\delta`.

    Parameters
    ----------
    perpendicular_flow_nusselt_number:
        :math:`\text{Nu}_{90}`. The perpendicular flow Nusselt number.
    angle_of_attack:
        :math:`\delta~\left[\text{radian}\right]`. The wind angle-of-attack.
    conductor_roughness:
        :math:`\text{Rs}`. The roughness number

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Nu}_\delta`. The Nusselt number for the given wind angle-of-attack.
    """
    return _correct_wind_direction_effect_on_nusselt_number(
        perpendicular_flow_nusselt_number,
        angle_of_attack,
        conductor_roughness,
    )


## Natural convection computations (no wind):
#############################################


def _check_horizontal_natural_nusselt_number(
    grashof_number: Unitless, prandtl_number: Unitless
) -> None:
    GrPr = grashof_number * prandtl_number
    if np.any(GrPr < 0):
        raise ValueError("GrPr cannot be negative.")
    elif np.any(GrPr > 1e12):
        raise ValueError("GrPr out of bounds: Must be < 10^12.")


@vectorize(nopython=True)
def _compute_horizontal_natural_nusselt_number(
    grashof_number: Unitless,
    prandtl_number: Unitless,
) -> Unitless:
    GrPr = grashof_number * prandtl_number

    if GrPr < 1e-1:
        return 0
    elif GrPr < 1e2:
        return 1.020 * GrPr**0.148
    elif GrPr < 1e4:
        return 0.850 * GrPr**0.188
    elif GrPr < 1e7:
        return 0.480 * GrPr**0.250
    else:
        return 0.125 * GrPr**0.333


def compute_horizontal_natural_nusselt_number(
    grashof_number: Unitless,
    prandtl_number: Unitless,
) -> Unitless:
    r"""The Nusselt number for natural (passive) convection on a horizontal conductor.

    Equation (23) and Table 5 on pages 27-28 of :cite:p:`cigre601`.

    The natural convection Nusselt number is denoted by both :math:`\text{Nu}_\text{nat}`
    and :math:`\text{Nu}_0` (due to the conductor declination being :math:`0^\circ`)
    in :cite:p:`cigre601`.

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


def _check_conductor_inclination(
    conductor_inclination: Radian,
    conductor_roughness: Unitless,
) -> None:
    beta = np.degrees(conductor_inclination)
    Rs = conductor_roughness

    smooth_mask = np.logical_or(Rs == 0, np.isnan(Rs))
    smooth_out_of_bounds = np.logical_and(smooth_mask, beta > 60)
    rough_out_of_bounds = np.logical_and(np.logical_not(smooth_mask), beta > 80)
    out_of_bounds = np.logical_or(smooth_out_of_bounds, rough_out_of_bounds)
    if np.any(out_of_bounds):
        raise ValueError(
            "Inclination must be < 60° for smooth conductors and 80° for stranded conductors"
        )


@vectorize(nopython=True)
def _correct_natural_nusselt_number_inclination(
    horizontal_natural_nusselt_number: Unitless,
    conductor_inclination: Radian,
    conductor_roughness: Unitless,
) -> Unitless:
    beta = np.degrees(conductor_inclination)
    Nu_nat = horizontal_natural_nusselt_number
    Rs = conductor_roughness

    if Rs == 0 or np.isnan(Rs):
        return Nu_nat * (1 - 1.58e-4 * beta**1.5)
    else:
        return Nu_nat * (1 - 1.76e-6 * beta**2.5)


def correct_natural_nusselt_number_inclination(
    horizontal_natural_nusselt_number: Unitless,
    conductor_inclination: Radian,
    conductor_roughness: Unitless,
) -> Unitless:
    r"""Correct the natural Nusselt number for the effect of the span inclination.

    Equation (24) on page 28 of :cite:p:`cigre601`.

    Parameters
    ----------
    horizontal_natural_nusselt_number:
        :math:`\text{Nu}_0`. The natural convection nusselt number assuming horizontal conductor.
    conductor_inclination:
        :math:`\beta~\left[\text{radian}\right]`. The inclination angle of the conductor. The
        inclination can be computed as
        :math:`\beta = \text{arctan2}\left(\left|y_1 - y_0\right|, L\right)`, where :math:`y_0`
        and :math:`y_1` are the altitude of the span endpoints (towers) and :math:`L` is the
        length of the span .
    conductor_roughness:
        :math:`Rs`. The roughness number.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Nu}_\beta`. The natural convection nusselt number where the conductor
        inclination is taken into account.
    """
    _check_conductor_inclination(conductor_inclination, conductor_roughness)
    return _correct_natural_nusselt_number_inclination(
        horizontal_natural_nusselt_number,
        conductor_inclination,
        conductor_roughness,
    )


def compute_nusselt_number(
    forced_convection_nusselt_number: Unitless,
    natural_nusselt_number: Unitless,
) -> Unitless:
    r"""Compute the nusselt number.

    Described in the text on p. 28 of :cite:p:`cigre601`.

    Parameters
    ----------
    forced_convection_nusselt_number:
        :math:`\text{Nu}_\delta`. The Nusselt number for the given wind angle-of-attack.
    natural_nusselt_number:
        :math:`\text{Nu}_\delta`. The natural convection nusselt number where the conductor
        inclination is taken into account.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`Nu`. The nusselt number.
    """
    return np.maximum(forced_convection_nusselt_number, natural_nusselt_number)


# Cooling computation
#####################


def compute_convective_cooling(
    surface_temperature: Celsius,
    air_temperature: Celsius,
    nusselt_number: Unitless,
    thermal_conductivity_of_air: WattPerMeterPerKelvin,
) -> WattPerMeter:
    r"""Compute the convective cooling of the conductor.

    Equation (17) on page 24 of :cite:p:`cigre601`.

    Parameters
    ----------
    surface_temperature:
        :math:`T_s~\left[^\circ\text{C}\right]`. The conductor surface temperature.
    air_temperature:
        :math:`T_a~\left[^\circ\text{C}\right]`. The ambient air temperature.
    nusselt_number:
        :math:`Nu`. The nusselt number.
    thermal_conductivity_of_air:
        :math:`\lambda_f~\left[\text{W}~\text{m}^{-1}~\text{K}^{-1}\right]`. The thermal
        conductivity of air at the given temperature.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`P_c~\left[\text{W}~\text{m}^{-1}\right]`. The convective cooling of the conductor.
        Either due to wind, or passive convection, whichever is largest.
    """
    pi = np.pi
    lambda_f = thermal_conductivity_of_air
    T_s = surface_temperature
    T_a = air_temperature
    Nu = nusselt_number

    return pi * lambda_f * (T_s - T_a) * Nu
