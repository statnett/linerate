import numpy as np
from numpy import pi

from ...units import Meter, Unitless, WattPerSquareMeter
from .. import cigre601


def compute_direct_solar_radiation(
    sin_solar_altitude: Unitless,
    height_above_sea_level: Meter,
) -> WattPerSquareMeter:
    r"""Compute the direct solar radiation.

    On page 19 of :cite:p:`cigre601`. Equation (10) states that the direct solar
    radiation on a surface normal to the solar beam at sea level, :math:`I_{B(0)}`, is given by

    .. math::

        N_s \frac{1280 \sin(H_s)}{\sin(H_s) + 0.314},

    where :math:`H_s` is the solar altitude.
    To correct for height above sea level, we use the Eq. 19 from Cigre 601,
    since no equation is provided in Cigre 207.

    Parameters
    ----------
    sin_solar_altitude:
        :math:`\sin\left(H_s\right)`. The sine of the solar altitude.
    height_above_sea_level:
        :math:`y~\left[\text{m}\right]`. The conductor's altitude.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`I_D~\left[\text{W}~\text{m}^{-2}\right]`. The direct solar radiation.
    """
    clearness_ratio = 1.0
    return cigre601.solar_heating.compute_direct_solar_radiation(
        sin_solar_altitude, clearness_ratio, height_above_sea_level
    )


def compute_diffuse_sky_radiation(
    direct_solar_radiation: WattPerSquareMeter,
    sin_solar_altitude: Unitless,
) -> WattPerSquareMeter:
    r"""Compute the diffuse radiation (light scattered in the atmosphere).

    On page 38 of :cite:p:`cigre207`.

    This equation differ from :cite:p:`cigre601`, however the difference is small, and the
    diffuse radiation is a small contributor to the overall solar radiation, so the total
    discrepancy between the models is small.

    Parameters
    ----------
    direct_solar_radiation:
        :math:`I_D~\left[\text{W}~\text{m}^{-2}\right]`. The direct solar radiation.
    sin_solar_altitude:
        :math:`\sin\left(H_s\right)`. The sine of the solar altitude.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`I_d~\left[\text{W}~\text{m}^{-2}\right]`.The diffuse solar radiation.
    """
    sin_H_s = sin_solar_altitude
    I_D = direct_solar_radiation
    return np.maximum(0, (570 - 0.47 * I_D)) * np.maximum(0, sin_H_s) ** 1.2


def compute_global_radiation_intensity(
    direct_solar_radiation: WattPerSquareMeter,
    diffuse_sky_radiation: WattPerSquareMeter,
    albedo: Unitless,
    sin_angle_of_sun_on_line: Unitless,
    sin_solar_altitude: Unitless,
) -> WattPerSquareMeter:
    r"""Compute the global radiation intensity experienced by the conductor.

    Equation (47) on page 38 of :cite:p:`cigre207` state that the global radiation intensity,
    :math:`I_T`, is given by

    .. math::

        I_T =
            I_D \left(\sin(\eta) + 0.5 F \pi \sin(H_s)\right) +
            0.5 I_d \pi \left(1 + F \right),

    where :math:`\eta` is the incidence angle of the sun on the line, :math:`H_s` is the solar
    altitude and :math:`F` is the ground albedo (amount of radiation diffusely reflected from the
    ground). The factor :math:`0.5 \pi` is due the assumption that the ground reflects light
    diffusely and uniformly in all directions, so the reflected energy is always directed normally
    to the line. In CIGRE207, it is also assumed that the diffuse radiation is uniformly directed,
    which leads to :math:`I_d \pi/2 (1 + F)` instead of  :math:`I_d (1 + 0.5 F \pi)`

    Parameters
    ----------
    direct_solar_radiation:
        :math:`I_D~\left[\text{W}~\text{m}^{-2}\right]`. The direct solar radiation.
    diffuse_sky_radiation:
        :math:`I_d~\left[\text{W}~\text{m}^{-2}\right]`.The diffuse solar radiation.
    albedo:
        :math:`F`. The ground albedo.
    sin_angle_of_sun_on_line:
        :math:`\sin\left(\eta\right)`. The sine of the angle of the sun on the line.
    sin_solar_altitude:
        :math:`\sin\left(H_s\right)`. The sine of the solar altitude.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`I_T~\left[\text{W}~\text{m}^{-2}\right]`. The global radiation intensity.

    Note
    ----
    The following values are given for the albedo in :cite:p:`cigre601`:

    .. list-table::
        :widths: 50 50
        :header-rows: 1

        * - Ground
          - Albedo
        * - Water (:math:`H_s > 30^\circ`)
          - 0.05
        * - Forest
          - 0.1
        * - Urban areas
          - 0.15
        * - Soil, grass and crops
          - 0.2
        * - Sand
          - 0.3
        * - Ice
          - 0.4-0.6
        * - Snow
          - 0.6-0.8
    """
    I_D = direct_solar_radiation
    I_d = diffuse_sky_radiation
    F = albedo
    sin_H_s = sin_solar_altitude
    sin_eta = sin_angle_of_sun_on_line
    F_pi_half = 0.5 * pi * F

    return I_D * (sin_eta + F_pi_half * sin_H_s) + I_d * pi / 2 * (1 + F)  # type: ignore
