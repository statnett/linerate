import numpy as np
from numpy import pi

from ...units import Meter, Unitless, WattPerMeter, WattPerSquareMeter


def compute_direct_solar_radiation(
    sin_solar_altitude: Unitless,
    clearness_ratio: Unitless,
    height_above_sea_level: Meter,
) -> WattPerSquareMeter:
    r"""Compute the direct solar radiation.

    Equation (10-11) on page 19 of :cite:p:`cigre601`. Equation (10) states that the direct solar
    radiation on a surface normal to the solar beam at sea level, :math:`I_{B(0)}`, is given by

    .. math::

        N_s \frac{1280 \sin(H_s)}{\sin(H_s) + 0.314},

    where :math:`N_s` is the clearness ratio which is used to adjust the amount of radiation
    compared to what goes through a standard Indian atmosphere, and :math:`H_s` is the solar
    altitude.

    While the solar radiation model is based on :cite:p:`sharma1965interrelationships` and
    therefore have parameters estimated for an Indian atmosphere, it gives comparable results to
    the solar radiation model in the IEEE standard :cite:p:`ieee738`. It is therefore reasonable to
    assume that the parameters work in other climates as well.

    Parameters
    ----------
    sin_solar_altitude:
        :math:`\sin\left(H_s\right)`. The sine of the solar altitude.
    clearness_ratio:
        :math:`N_s`. The clearness ratio (or clearness number in
        :cite:p:`sharma1965interrelationships,cigre207`).
    height_above_sea_level:
        :math:`y~\left[\text{m}\right]`. The conductor's altitude.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`I_B~\left[\text{W}~\text{m}^{-2}\right]`. The direct solar radiation.

    Note
    ----
    The 1280 originates and 0.314 in the above equation originates from
    :cite:p:`sharma1965interrelationships`, which is cited in :cite:p:`morgan1982thermal` (which is
    listed as the reference in :cite:p:`cigre601`). In :cite:p:`sharma1965interrelationships` the
    empirical relationship

    .. math::

        I_{B(0)} = \frac{1.842 \sin(H_s)}{\sin(H_s) + 0.3135}~\text{Ly}~\text{min}^{-1}

    is introduced, and by converting from Langley per minute to :math:`\text{W}~\text{m}^{-2}`, we
    obtain

    .. math::

        I_{B(0)} = N_s \frac{1284.488 \sin(H_s)}{\sin(H_s) + 0.3135}~\text{W}~\text{m}^{-2},

    which is equal to the equation we use (with three significant digits).
    """
    sin_H_s = sin_solar_altitude
    N_s = clearness_ratio
    y = height_above_sea_level

    I_B_0 = N_s * 1280 * sin_H_s / (sin_H_s + 0.314)
    # Equation 19 says that
    # I_B = I_B_0 * (1 + 1.4e-4 * y * (1367/I_B_0 - 1)) # noqa: E800
    # However, if I_B_0 = 0, this will divide by 0. To return NaN-values if and only
    # if the input is NaN, we therefore reformulate it
    scaled_y = 1.4e-4 * y
    I_B = I_B_0 * (1 - scaled_y) + 1367 * scaled_y

    return np.where(sin_H_s >= 0, I_B, 0 * I_B)


def compute_diffuse_sky_radiation(
    direct_solar_radiation: WattPerSquareMeter,
    sin_solar_altitude: Unitless,
) -> WattPerSquareMeter:
    r"""Compute the diffuse radiation (light scattered in the atmosphere).

    Equation (13) on page 20 of :cite:p:`cigre601`.

    This equation differ from :cite:p:`cigre207`, however the difference is small, and the
    diffuse radiation is a small contributor to the overall solar radiation, so the total
    discrepancy between the models is small.

    Parameters
    ----------
    direct_solar_radiation:
        :math:`I_B~\left[\text{W}~\text{m}^{-2}\right]`. The direct solar radiation.
    sin_solar_altitude:
        :math:`\sin\left(H_s\right)`. The sine of the solar altitude.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`I_d~\left[\text{W}~\text{m}^{-2}\right]`.The diffuse solar radiation.
    """
    sin_H_s = sin_solar_altitude
    I_B = direct_solar_radiation
    return np.maximum(0, (430.5 - 0.3288 * I_B)) * np.maximum(0, sin_H_s)


def compute_global_radiation_intensity(
    direct_solar_radiation: WattPerSquareMeter,
    diffuse_sky_radiation: WattPerSquareMeter,
    albedo: Unitless,
    sin_angle_of_sun_on_line: Unitless,
    sin_solar_altitude: Unitless,
) -> WattPerSquareMeter:
    r"""Compute the global radiation intensity experienced by the conductor.

    Equation (9) on page 18 of :cite:p:`cigre601` state that the global radiation intensity,
    :math:`I_T`, is given by

    .. math::

        I_T =
            I_B \left(\sin(\eta) + 0.5 F \pi \sin(H_s)\right) +
            I_d \left(1 + 0.5 F \pi\right),

    where :math:`\eta` is the incidence angle of the sun on the line, :math:`H_s` is the solar
    altitude and :math:`F` is the ground albedo (amount of radiation diffusely reflected from the
    ground). The factor :math:`0.5 \pi` is due the assumption that the ground reflects light
    diffusely and uniformly in all directions, so the reflected energy is always directed normally
    to the line. In CIGRE207, it is also assumed that the diffuse radiation is uniformly directed,
    which leads to :math:`I_d (0.5 \pi + 0.5 F \pi)` instead of  :math:`I_d (1 + 0.5 F \pi)`

    Parameters
    ----------
    direct_solar_radiation:
        :math:`I_B~\left[\text{W}~\text{m}^{-2}\right]`. The direct solar radiation.
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
    r"""Compute the solar heating experienced by the conductor.

    Equation (8) on page 18 of :cite:p:`cigre601`.

    Parameters
    ----------
    absorptivity:
        :math:`\alpha_s`. Material constant. According to :cite:p:`cigre601`, it starts at
        approximately 0.2 for new cables and reaches a constant value of approximately 0.9
        after about one year.
    global_radiation_intensity:
        :math:`I_T~\left[\text{W}~\text{m}^{-2}\right]`.The global radiation intensity.
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`P_S~\left[\text{W}~\text{m}^{-1}\right]`. The solar heating of the conductor
    """
    alpha_s = absorptivity
    I_T = global_radiation_intensity
    D = conductor_diameter

    return alpha_s * I_T * D
