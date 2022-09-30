from textwrap import dedent

import numpy as np
from scipy.interpolate import interp1d

from ..units import (
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
    """Return the difference between the core and surface temperature. Equation 15-16 on page 22.

    The conductor thermal conductivity is between 0.5 and 7 W/(m k). Recommended values are
    0.7 W/(m k) for conductors with no tension on the aluminium strands and 1.5 W/(m k) for
    with aluminium strands under a tension of at least 40 N.
    """
    # TODO: Check original source.
    # I'm not sure I understand how the temperature gradient can be independent of diameter for
    # full-body mono-metallic conductors. Possibly some assumption with small R?
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
    """Equation 18 on page 24.

    This polynomial fit is compared with eight values from a textbook: TODO WHICH TEXTBOOK
    which showed a maximum relative error of 0.007.
    """
    T_f = film_temperature
    return 2.368e-2 + 7.23e-5 * T_f - 2.763e-8 * (T_f**2)


def compute_air_density(
    film_temperature: Celsius, height_above_sea_level: Meter
) -> KilogramPerCubeMeter:
    """Equation 20 on page 25."""
    T_f = film_temperature
    y = height_above_sea_level
    return (1.293 - 1.525e-4 * y + 6.379e-9 * (y**2)) / (1 + 0.00367 * T_f)


def compute_dynamic_viscosity_of_air(film_temperature: Celsius) -> KilogramPerMeterPerSecond:
    """Equation 19 on page 25."""
    T_f = film_temperature
    return 17.239e-6 + 4.635e-8 * T_f - 2.03e-11 * (T_f**2)


def compute_kinematic_viscosity_of_air(
    dynamic_viscosity_of_air: KilogramPerMeterPerSecond, air_density: KilogramPerCubeMeter
) -> SquareMeterPerSecond:
    """Definition on page 25."""
    return dynamic_viscosity_of_air / air_density


# Unitless quantities
#####################


def compute_reynolds_number(
    wind_speed: MeterPerSecond,
    conductor_diameter: Meter,
    kinematic_viscosity_of_air: SquareMeterPerSecond,
) -> Unitless:
    """Definition of Reynolds number using the conductor diameter as characteristic length scale."""
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
    """Compute the Grashof number. Defined in the nomenclature on page 7"""
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
    """Compute the Prandlt number. Defined in the nomenclature on page 8."""
    lambda_f = thermal_conductivity_of_air
    mu_f = dynamic_viscosity_of_air
    c_f = specific_heat_capacity_of_air

    return c_f * mu_f / lambda_f


def compute_conductor_roughness(
    conductor_diameter: Meter,
    outer_layer_strand_diameter: Meter,
) -> Unitless:
    """Compute the surface roughness of the conductor. defined in the text on page 25."""
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

_smooth_conductor_nu90_coefficients = interp1d(
    np.array([0, 35.0, 5_000.0, 50_000.0, 200_000.0]),
    np.array(
        [
            [0, 0],
            [0.583, 0.471],
            [0.148, 0.633],
            [0.0208, 0.814],
            [0.0208, 0.814],
        ]
    ),
    kind="previous",
    axis=0,
)

_smooth_stranded_conductor_nu90_coefficients = interp1d(
    np.array(
        [
            0,
            100.0,
            2_650.0,
            50_000.0,
        ]
    ),
    np.array(
        [
            [0, 0],
            [0.641, 0.471],
            [0.178, 0.633],
            [0.178, 0.633],
        ]
    ),
    kind="previous",
    axis=0,
)

_rough_stranded_conductor_nu90_coefficients = interp1d(
    np.array(
        [
            0,
            100.0,
            2_650.0,
            50_000.0,
        ]
    ),
    np.array(
        [
            [0, 0],
            [0.641, 0.471],
            [0.048, 0.800],
            [0.048, 0.800],
        ]
    ),
    kind="previous",
    axis=0,
)


def compute_perpendicular_flow_nusseltnumber(
    reynolds_number: Unitless,
    conductor_roughness: Meter,
) -> Unitless:
    """Compute the nusselt number for perpendicular flow (equation 21 and Table 4 on pages 25-26).

    The perpendicular flow nusselt number is denoted as Nu_90 in CIGRE TB 601 since the wind's
    angle of attack is 90 degrees.

    The Nusselt number is the ratio of conductive heat transfer to convective heat transfer.
    """
    # TODO: Look at references for this table
    # TODO: Possibly use numba vectorize instead
    Re = reynolds_number
    Rs = conductor_roughness

    if Rs == 0 or np.isnan(Rs):
        nu_90_coeffs = _smooth_conductor_nu90_coefficients(Re)
    elif Rs <= 0.05:
        nu_90_coeffs = _smooth_stranded_conductor_nu90_coefficients(Re)
    else:
        nu_90_coeffs = _rough_stranded_conductor_nu90_coefficients(Re)

    B = nu_90_coeffs[..., 0]
    n = nu_90_coeffs[..., 1]
    return B * Re**n  # type: ignore


def correct_wind_direction_effect_on_nusselt_number(
    perpendicular_wind_nusselt_number: Unitless,
    angle_of_attack: Radian,
    conductor_roughness: Unitless,
) -> Unitless:
    """Compute the nusselt number for perpendicular flow (equation 21 and Table 4 on pages 25-26).

    The perpendicular flow nusselt number is denoted as Nu_delta in CIGRE TB 601 since the wind's
    angle of attack is delta.
    """
    delta = angle_of_attack
    Nu_90 = perpendicular_wind_nusselt_number
    Rs = conductor_roughness

    sin_delta = np.sin(delta)

    if Rs == 0 or np.isnan(Rs):
        sin_delta_sq = sin_delta**2
        cos_delta_sq = 1 - sin_delta_sq

        correction_factor = (sin_delta_sq + cos_delta_sq * 0.0169) ** 0.225
    else:
        correction_factor = np.where(
            delta <= np.radians(24),
            0.42 + 0.68 * (sin_delta**1.08),
            0.42 + 0.58 * (sin_delta**0.90),
        )

    return correction_factor * Nu_90


## Natural convection computations (no wind):
#############################################


_nu_0_coefficients = interp1d(
    [1e-1, 1e2, 1e4, 1e7, 1e12],
    [
        [1.020, 0.148],
        [0.850, 0.188],
        [0.480, 0.250],
        [0.125, 0.333],
        [0.125, 0.333],
    ],
    kind="previous",
    axis=0,
)  # TODO: Possibly use sympy or numba vectorize instead


def compute_horizontal_natural_nusselt_number(
    grashof_number: Unitless,
    prandtl_number: Unitless,
) -> Unitless:
    """Equation 23 and Table 5 on pages 27-28."""
    GrPr = grashof_number * prandtl_number
    nu_0_coefficients = _nu_0_coefficients(GrPr)
    A = nu_0_coefficients[..., 0]
    m = nu_0_coefficients[..., 1]
    return A * (GrPr**m)  # type: ignore


def correct_natural_nusselt_number_inclination(
    horizontal_natural_nusselt_number: Unitless,
    conductor_inclination: Radian,
    conductor_roughness: Unitless,
) -> Unitless:
    """Equation 24 on page 28"""
    beta = np.degrees(conductor_inclination)
    Nu_nat = horizontal_natural_nusselt_number
    Rs = conductor_roughness

    if Rs == 0 or np.isnan(Rs):
        if np.any(beta > 60):
            raise ValueError(
                f"Inclination must be less than 60° for smooth conductors (it is {beta})"
            )

        return Nu_nat * (1 - 1.58e-4 * beta**1.5)
    else:
        if np.any(beta > 80):
            raise ValueError(
                f"Inclination must be less than 80° for smooth conductors (it is {beta})"
            )

        return Nu_nat * (1 - 1.76e-6 * beta**2.5)


def compute_nusselt_number(
    forced_convection_nusselt_number: Unitless,
    natural_nusselt_number: Unitless,
) -> Unitless:
    """Compute the nusselt number (p. 28)"""
    return np.maximum(forced_convection_nusselt_number, natural_nusselt_number)


# Cooling computation
#####################


def compute_convective_cooling(
    surface_temperature: Celsius,
    air_temperature: Celsius,
    nusselt_number: Unitless,
    thermal_conductivity_of_air: WattPerMeterPerKelvin,
) -> WattPerMeter:
    """Equation 17 on page 24."""
    pi = np.pi
    lambda_f = thermal_conductivity_of_air
    T_s = surface_temperature
    T_a = air_temperature
    Nu = nusselt_number

    return pi * lambda_f * (T_s - T_a) * Nu
