import numpy as np

from linerate.units import Celsius, Unitless, WattPerMeter, WattPerMeterPerKelvin


def compute_convective_cooling(
    surface_temperature: Celsius,
    air_temperature: Celsius,
    nusselt_number: Unitless,
    thermal_conductivity_of_air: WattPerMeterPerKelvin,
) -> WattPerMeter:
    r"""Compute the convective cooling of the conductor.

    Equation (17) on page 24 of :cite:p:`cigre601`
    and Equation (12) on page 6 of :cite:p:`cigre207`.

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
