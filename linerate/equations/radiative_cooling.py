import numpy as np
from scipy.constants import Stefan_Boltzmann as stefan_boltzmann_constant

from ..units import Celsius, Meter, Unitless, WattPerMeter


def compute_radiative_cooling(
    surface_temperature: Celsius,
    ambient_temperature: Celsius,
    conductor_diameter: Meter,
    conductor_emissivity: Unitless,
) -> WattPerMeter:
    """Equation 27 on page 30."""
    sigma_B = stefan_boltzmann_constant
    pi = np.pi
    D = conductor_diameter
    T_s = surface_temperature
    T_a = ambient_temperature
    epsilon_s = conductor_emissivity

    return pi * D * sigma_B * epsilon_s * ((T_s + 273.15) ** 4 - (T_a + 273.15) ** 4)  # type: ignore # noqa
