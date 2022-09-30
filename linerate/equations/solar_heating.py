from typing import Tuple
from warnings import warn

import numpy as np
from numpy import cos, pi, radians, sin
from pysolar.solar import (  # TODO: pysolar uses GPL, maybe library with less restrictive lisence
    get_altitude_fast,
    get_azimuth_fast,
)

from ..units import Date, Degrees, Meter, Radian, Unitless, WattPerMeter, WattPerSquareMeter
from .math import switch_cos_sin


def compute_solar_time_of_day(longitude: Unitless, utc_time_in_hours: int) -> Radian:
    """From page 19:
    To obtain solar time, add 4 minutes per degree of longitude east of standard time,
    or subtract 4 minutes per degree west of standard time.

    Since 4 minutes is due to 24*60=1440 minutes in a day / 360 degrees = 4 minutes per degree.
    """
    return (utc_time_in_hours + longitude * (24 / (2 * pi))) % 24


def compute_declination(day_of_year: int) -> Radian:
    """From page 19."""
    return radians(23.3) * sin((2 * pi / 365) * (284 + day_of_year))  # type: ignore


def compute_sin_solar_angles(
    latitude: Degrees,
    longitude: Degrees,
    when: Date,
    conductor_azimuth: Radian,
) -> Tuple[Unitless, Unitless]:
    """Equation (12) and (14) on page 19 and 20.

    The solar altitude and azimuth is computed with pysolar instead of following the equations in
    CIGRE TB 601, since the report has some sign problems.
    """
    warn(
        "We need to double check that the conductor azimuth and the solar azimuth are aligned correctly."  # noqa
    )
    gamma_c = conductor_azimuth
    gamma_s = np.radians(get_azimuth_fast(latitude, longitude, when))
    H_s = np.radians(get_altitude_fast(latitude, longitude, when))

    cos_eta = cos(H_s) * cos(gamma_s - gamma_c)
    sin_eta = switch_cos_sin(cos_eta)
    sin_H_s = np.sin(H_s)

    sin_solar_altitude = sin_H_s
    sin_angle_of_sun_on_line = sin_eta
    return sin_solar_altitude, sin_angle_of_sun_on_line


def compute_direct_solar_radiation(
    sin_solar_altitude: Unitless,
    clearness_ratio: Unitless,
    height_above_sea_level: Meter,
) -> WattPerSquareMeter:
    """Equation (10-11) on page 19."""
    sin_H_s = sin_solar_altitude
    N_s = clearness_ratio
    y = height_above_sea_level

    I_B_0 = N_s * 1280 * sin_H_s / (sin_H_s + 0.314)
    # Equation 19 says that
    # I_B = I_B_0 * (1 + 1.4e-4 * y * (1367/I_B_0 - 1))
    # However, if I_B_0 = 0, this will divide by 0. To return NaN-values if and only
    # if the input is NaN, we therefore reformulate it
    scaled_y = 1.4e-4 * y
    I_B = I_B_0 * (1 - scaled_y) + 1367 * scaled_y

    return np.where(I_B >= 0, I_B, 0 * I_B)  # To keep NaN values


def compute_diffuse_sky_radiation(
    direct_solar_radiation: WattPerSquareMeter,
    sin_solar_altitude: Unitless,
) -> WattPerSquareMeter:
    """Equation 13 on page 20"""
    sin_H_s = sin_solar_altitude
    I_B = direct_solar_radiation
    return (430.5 - 0.3288 * I_B) * sin_H_s


def compute_global_radiation_intensity(
    direct_solar_radiation: WattPerSquareMeter,
    diffuse_sky_radiation: WattPerSquareMeter,
    albedo: Unitless,
    sin_angle_of_sun_on_line: Unitless,
    sin_solar_altitude: Unitless,
) -> WattPerSquareMeter:
    """Equation (9) on page 18."""
    I_B = direct_solar_radiation
    I_d = diffuse_sky_radiation
    F = albedo
    sin_H_s = sin_solar_altitude
    sin_eta = sin_angle_of_sun_on_line
    F_pi_half = 0.5 * pi * F

    return I_B * (sin_eta + F_pi_half * sin_H_s) + I_d * (1 + F_pi_half)  # type: ignore


def compute_solar_heating(
    absorptivity: Unitless,
    global_radiation_intensity: WattPerSquareMeter,
    conductor_diameter: Meter,
) -> WattPerMeter:
    """Equation (8) on page 18.

    :param absorptivity: Material constant based on how rough the line is.
        Starts at approximately 0.2 for new cables and reaches its constant value
        of 0.9 after about one year.
    :param global_radiation_intensity:
    :param conductor_diameter:
    :return:
    """
    alpha_s = absorptivity
    I_T = global_radiation_intensity
    D = conductor_diameter

    return alpha_s * I_T * D
