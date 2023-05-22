import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx
from scipy.constants import Stefan_Boltzmann as stefan_boltzmann_constant

import linerate.equations.radiative_cooling as radiative_cooling


@hypothesis.given(conductor_emissivity=st.floats(allow_nan=False))
def test_radiative_cooling_scales_affinely_with_emissivity(conductor_emissivity):
    # Set parameters so 2piR (T_s^4 - T_a^4) = 1
    D = 1 / np.pi
    epsilon = conductor_emissivity
    sigma = stefan_boltzmann_constant
    T_s = -272.15
    T_a = -273.15

    assert radiative_cooling.compute_radiative_cooling(T_s, T_a, D, epsilon) == approx(
        sigma * epsilon
    )


@hypothesis.given(conductor_diameter=st.floats(allow_nan=False, min_value=0, max_value=1e10))
def test_radiative_cooling_scales_affinely_with_diameter(conductor_diameter):
    # Set parameters so pisigma (T_s^4 - T_a^4) = 1
    D = conductor_diameter
    epsilon = 1 / np.pi
    sigma = stefan_boltzmann_constant
    T_s = -272.15
    T_a = -273.15

    assert radiative_cooling.compute_radiative_cooling(T_s, T_a, D, epsilon) == approx(sigma * D)


@hypothesis.given(surface_temperature=st.floats(allow_nan=False, min_value=-273.15, max_value=1e10))
def test_radiative_cooling_scales_power_four_with_surface_temperature(surface_temperature):
    # Set parameters so 2 pi sigma epsilon = 1
    D = 1 / np.pi
    sigma = stefan_boltzmann_constant
    epsilon = 1 / sigma
    T_s = surface_temperature
    T_a = -273.15

    assert radiative_cooling.compute_radiative_cooling(T_s, T_a, D, epsilon) == approx(
        (T_s + 273.15) ** 4
    )


@hypothesis.given(air_temperature=st.floats(allow_nan=False, min_value=-273.15, max_value=1e10))
def test_radiative_cooling_scales_power_four_with_air_temperature(air_temperature):
    # Set parameters so 2 pi sigma epsilon = 1
    D = 1 / np.pi
    sigma = stefan_boltzmann_constant
    epsilon = 1 / sigma
    T_s = -273.15
    T_a = air_temperature

    assert radiative_cooling.compute_radiative_cooling(T_s, T_a, D, epsilon) == approx(
        -((T_a + 273.15) ** 4)
    )


@pytest.mark.parametrize(
    "conductor_diameter, conductor_emissivity, surface_temperature, air_temperature, cooling",
    [
        (1 / np.pi, 1 / stefan_boltzmann_constant, -273.15, -273.15, 0),
        (1 / np.pi, 1 / stefan_boltzmann_constant, -272.15, -273.15, 1),
        (1 / np.pi, 1 / stefan_boltzmann_constant, -268.15, -273.15, 5**4),
        (1 / np.pi, 1 / stefan_boltzmann_constant, -268.15, -272.15, 5**4 - 1),
        (1 / np.pi, 1 / stefan_boltzmann_constant, -268.15, -269.15, 5**4 - 4**4),
        (2 / np.pi, 1 / stefan_boltzmann_constant, -268.15, -269.15, 2 * (5**4 - 4**4)),
        (2 / np.pi, 1, -268.15, -269.15, 2 * stefan_boltzmann_constant * (5**4 - 4**4)),
    ],
)
def test_radiative_cooling_with_example(
    conductor_diameter, conductor_emissivity, surface_temperature, air_temperature, cooling
):
    D = conductor_diameter
    epsilon = conductor_emissivity
    T_s = surface_temperature
    T_a = air_temperature

    assert radiative_cooling.compute_radiative_cooling(T_s, T_a, D, epsilon) == approx(cooling)
