import numpy as np
from scipy.constants import Stefan_Boltzmann as stefan_boltzmann_constant

from ..units import Celsius, Meter, Unitless, WattPerMeter


def compute_radiative_cooling(
    surface_temperature: Celsius,
    air_temperature: Celsius,
    conductor_diameter: Meter,
    conductor_emissivity: Unitless,
) -> WattPerMeter:
    r"""Compute the radiative cooling due to black body radiation.

    Equation (27) on page 30 of :cite:p:`cigre601`.

    Parameters
    ----------
    surface_temperature:
        :math:`T_s~\left[^\circ\text{C}\right]`. The conductor surface temperature.
    air_temperature:
        :math:`T_a~\left[^\circ\text{C}\right]`. The ambient air temperature.
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    conductor_emissivity:
        :math:`\epsilon_s`. The emmisivity of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`P_r~\left[\text{W}~\text{m}^{-1}\right]`. The radiative cooling of the conductor.
    """
    sigma_B = stefan_boltzmann_constant
    pi = np.pi
    D = conductor_diameter
    T_s = surface_temperature
    T_a = air_temperature
    epsilon_s = conductor_emissivity

    return pi * D * sigma_B * epsilon_s * ((T_s + 273.15) ** 4 - (T_a + 273.15) ** 4)  # type: ignore # noqa
