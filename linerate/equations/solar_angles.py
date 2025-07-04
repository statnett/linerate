import numpy as np
from numba import vectorize
from linerate.equations import math

from ..units import Date, Degrees, Radian, Unitless
from ..types import Span


def _get_day_of_year(when: Date) -> Unitless:
    YearResolutionType = np.datetime64(1, "Y")
    DayResolutionType = np.datetime64(1, "D")

    return (when.astype(DayResolutionType) - when.astype(YearResolutionType)).astype(float) + 1


def _get_hour_of_day(when: Date) -> Unitless:
    DayResolutionType = np.datetime64(1, "D")
    HourResolutionType = np.datetime64(1, "h")
    return (when.astype(HourResolutionType) - when.astype(DayResolutionType)).astype(float)


def _get_minute_of_hour(when: Date) -> Unitless:
    HourResolutionType = np.datetime64(1, "h")
    MinuteResolutionType = np.datetime64(1, "m")

    return (when.astype(MinuteResolutionType) - when.astype(HourResolutionType)).astype(float)


def compute_hour_angle_relative_to_noon(when: Date, longitude: Degrees) -> Radian:
    r"""Compute the hour angle.

    Described in the text on p. 18 of :cite:p:`ieee738`. The hour angle is the number of hours
    from noon times 15^\circ. This means that the hour angle for 11:00 is -15^\circ, and the
    hour angle for 14:00 is 30^\circ.

    This function converts the hour angle to radians by multiplying it by \frac{\pi}{12},
    which is the same as 15^\circ.

    The hour angle is used when calculating the solar altitude.
    This function does not take into account the difference between apparent/actual
    and mean solar time, which means that the result may be up to 15 minutes from the
    correct hour angle.

    Parameters
    ----------
    when:
        The time and date.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:'\omega~\left[\text{radian}\right]`. The hour angle relative to noon.
    """
    utc_hour = _get_hour_of_day(when)
    utc_minute = _get_minute_of_hour(when)
    pi = np.pi
    # We add longitude/15 since 15 degrees of longitude increases solar hour by 1
    return np.mod((-12 + utc_hour + utc_minute / 60 + longitude / 15), 24) * (
        pi / 12
    )  # pi/12 is 15 degrees


def compute_solar_declination(
    when: Date,
) -> Radian:
    r"""Compute the solar declination

    Equation (16b) on page 18 of :cite:p:`ieee738`.

    The function takes in a numpy.datetime64 object, and uses the _get_day_of_year
    function to find the day number of the year to be used to compute the solar declination.

    Parameters
    ----------
    when:
        The time and date.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\delta~\left[\text{radian}\right]`. The declination of the earth.
    """
    N = _get_day_of_year(when)
    return np.radians((23.3) * np.sin((284 + N) * 2 * np.pi / 365))


def compute_solar_azimuth_variable(
    latitude: Degrees,
    solar_declination: Radian,
    hour_angle_relative_to_noon: Radian,
) -> Radian:
    r"""Compute the solar azimuth variable.

    Equation (17b) on page 18 of :cite:p:`ieee738`.

    Parameters
    ----------
    latitude:
        :math:`Lat~\left[\text{degrees}\right]`. The latitude of the span (center).
    solar_declination:
        :math:`\delta~\left[\text{radian}\right]`. Solar declination (-23.45 to +23.45).
    hour_angle_relative_to_noon:
        :math:'\omega~\left[\text{radian}\right]`. The hour angle relative to noon.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\chi~\left[\text{radian}\right]`. The solar azimuth variable.
    """
    Lat = np.radians(latitude)
    delta = solar_declination
    omega = hour_angle_relative_to_noon
    return np.sin(omega) / (np.sin(Lat) * np.cos(omega) - np.cos(Lat) * np.tan(delta))


@vectorize
def _compute_solar_azimuth_constant(
    solar_azimuth_variable: Radian, hour_angle_relative_to_noon: Radian
) -> Radian:
    chi = solar_azimuth_variable
    omega = hour_angle_relative_to_noon
    pi = np.pi
    if -pi <= omega < 0:
        if chi >= 0:
            C = 0
        elif chi < 0:
            C = pi
    elif 0 <= omega < pi:
        if chi >= 0:
            C = pi
        elif chi < 0:
            C = 2 * pi
    return C


def compute_solar_azimuth_constant(
    solar_azimuth_variable: Radian, hour_angle_relative_to_noon: Radian
) -> Radian:
    r"""Compute the solar azimuth constant.

    Table 2 on page 18 of:cite:p:`ieee738`.

    Parameters
    ----------
    solar_azimuth_variable:
        :math:`\chi~\left[\text{radian}\right]`. The solar azimuth variable.
    hour_angle_relative_to_noon:
        :math:'\omega~\left[\text{radian}\right]`. The hour angle relative to noon.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`C~\left[\text{radian}\right]`. The solar azimuth constant.

    """
    return _compute_solar_azimuth_constant(solar_azimuth_variable, hour_angle_relative_to_noon)


def compute_solar_azimuth(
    solar_azimuth_constant: Radian,
    solar_azimuth_variable: Radian,
) -> Radian:
    r"""Compute the solar azimuth.

    Equation (17a) on page 18 of :cite:p:`ieee738`.

    Parameters
    ----------
    solar_azimuth_constant:
        :math:`C~\left[\text{radian}\right]`. The solar azimuth constant.
    solar_azimuth_variable:
        :math:`\chi~\left[\text{radian}\right]`. The solar azimuth variable.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`Z_c~\left[\text{radian}\right]`. The solar azimuth.
    """
    C = solar_azimuth_constant
    chi = solar_azimuth_variable
    return C + np.arctan(chi)


def compute_sin_solar_altitude(
    latitude: Degrees,
    solar_declination: Radian,
    hour_angle_relative_to_noon: Radian,
) -> Radian:
    r"""Compute the sine of the solar altitude

    This is an alteration of equation  (16a) on page 18 of :cite:p:`ieee738`.
    :math:`sin(H_c)` is calculated instead of :math:`H_c`.

    Parameters
    ----------
    latitude:
        :math:`Lat~\left[^\circ\right]`. The latitude of the span (center).
    ssolar_declination:
        :math:`\delta~\left[\text{radian}\right]`. Solar declination (-23.45 to +23.45).
    hour_angle_relative_to_noon:
        :math:'\omega~\left[\text{radian}\right]`. The hour angle relative to noon.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\sin\left(H_c\right)`. The sine of the solar altitude.
    """
    Lat = np.radians(latitude)
    delta = solar_declination
    omega = hour_angle_relative_to_noon
    return np.cos(Lat) * np.cos(delta) * np.cos(omega) + np.sin(Lat) * np.sin(delta)


def compute_cos_solar_effective_incidence_angle(
    sin_solar_altitude: Radian,
    solar_azimuth: Radian,
    conductor_azimuth: Radian,
) -> Radian:
    r"""Compute the cosine of the effective angle of incidence of the sun rays.

    This is an alteration of equation (9) on page 13 of :cite:p:`ieee738`.
    :math:`cos(\theta)` is calculated instead of :math:`\theta`.

    Parameters
    ----------
    sin_solar_altitude:
        :math:`sin(H_c)~\left[\text{radian}~\right]`. The sin of the solar altitude.
    solar_azimuth:
        :math:`Z_c~\left[\text{radian}~\right]`. The azimuth of the sun.
    conductor_azimuth:
        :math:`Z_l~\left[\text{radian}~\right]`. The azimuth of the conductor/line.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`cos(\theta)`. The cosine of the effective angle of incidence of the sun rays.
    """
    H_c = np.arcsin(sin_solar_altitude)
    Z_c = solar_azimuth
    Z_l = conductor_azimuth
    return np.cos(H_c) * np.cos(Z_c - Z_l)


def compute_sin_solar_altitude_for_span(span: Span, time: Date) -> Unitless:
    """Compute the sine of the solar altitude for a given span and time.

    This function computes the sine of the solar altitude at the midpoint of the span.
    It uses the latitude, longitude of the span to compute the
    hour angle and solar declination.

    Parameters
    ----------
    span:
        The span for which to compute the sine of the solar altitude.
    time:
        The time at which to compute the sine of the solar altitude.

    Returns
    -------
    Unitless
        The sine of the solar altitude at the midpoint of the span. (sin H_s)
    """
    phi = span.latitude
    delta = compute_solar_declination(time)
    omega = compute_hour_angle_relative_to_noon(time, span.longitude)
    return compute_sin_solar_altitude(phi, delta, omega)


def compute_sin_solar_effective_incidence_angle_for_span(
    span: Span, time: Date, sin_H_s: Unitless
) -> Unitless:
    """Compute the sine of the solar effective incidence angle for a given span and time.

    This function computes the sine of the solar effective incidence angle at the midpoint of the span.
    It uses the latitude, longitude, conductor azimuth, and solar altitude to compute the sine of the
    effective incidence angle.

    Parameters
    ----------
    span:
        The span for which to compute the sine of the solar effective incidence angle.
    time:
        The time at which to compute the sine of the solar effective incidence angle.
    sin_H_s:
        The sine of the solar altitude at the midpoint of the span
        (computed with `compute_sin_solar_altitude_for_span`).

    Returns
    -------
    Unitless
        The sine of the solar effective incidence angle at the midpoint of the span. (sin eta)
    """
    gamma_c = span.conductor_azimuth
    delta = compute_solar_declination(time)
    omega = compute_hour_angle_relative_to_noon(time, span.longitude)

    chi = compute_solar_azimuth_variable(span.latitude, delta, omega)
    C = compute_solar_azimuth_constant(chi, omega)
    gamma_s = compute_solar_azimuth(C, chi)  # Z_c in IEEE

    cos_eta = compute_cos_solar_effective_incidence_angle(sin_H_s, gamma_s, gamma_c)

    return math.switch_cos_sin(cos_eta)
