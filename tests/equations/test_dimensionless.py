# Tests for unitless quantities
###############################

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx

from linerate.equations import dimensionless


@hypothesis.given(wind_speed=st.floats(min_value=0, max_value=1000, allow_nan=False))
def test_reynolds_number_scales_linearly_with_wind_speed(wind_speed):
    V = wind_speed
    D = 1
    Nu_f = 1

    assert dimensionless.compute_reynolds_number(V, D, Nu_f) == pytest.approx(V)


@hypothesis.given(conductor_diameter=st.floats(min_value=0, max_value=1000, allow_nan=False))
def test_reynolds_number_scales_linearly_with_diameter(conductor_diameter):
    V = 1
    D = conductor_diameter
    Nu_f = 1

    assert dimensionless.compute_reynolds_number(V, D, Nu_f) == pytest.approx(D)


@hypothesis.given(kinematic_viscosity=st.floats(min_value=1e-10, allow_nan=False))
def test_reynolds_number_scales_inversly_with_kinematic_viscosity(kinematic_viscosity):
    V = 1
    D = 1
    Nu_f = kinematic_viscosity

    assert dimensionless.compute_reynolds_number(V, D, Nu_f) == approx(1 / Nu_f)


def test_reynolds_number_with_example():
    V = 0.1
    D = 1.2
    Nu_f = 10

    assert dimensionless.compute_reynolds_number(V, D, Nu_f) == approx(0.012)


@hypothesis.given(conductor_diameter=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False))
def test_grashof_number_scales_cubic_with_conductor_diameter(conductor_diameter):
    D = conductor_diameter
    T_s = 1
    T_a = 0
    Nu_f = 1
    g = 273.65  # 0.5*T_s in Kelvin

    Gr = dimensionless.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(D**3)


@hypothesis.given(surface_temperature=st.floats(min_value=273, max_value=1000, allow_nan=False))
def test_grashof_number_scales_correctly_with_surface_temperature(surface_temperature):
    D = 1
    T_s = surface_temperature
    T_a = 0
    Nu_f = 1
    g = 0.5 * T_s + 273.15

    Gr = dimensionless.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(abs(T_s))


@hypothesis.given(air_temperature=st.floats(min_value=273, max_value=1000, allow_nan=False))
def test_grashof_number_scales_correctly_with_air_temperature(air_temperature):
    D = 1
    T_s = 0
    T_a = air_temperature
    Nu_f = 1
    g = 0.5 * T_a + 273.15

    Gr = dimensionless.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(abs(T_a))


@hypothesis.given(
    kinematic_viscosity_of_air=st.floats(min_value=1e-5, max_value=1e10, allow_nan=False)
)
def test_grashof_number_scales_inversely_squared_with_kinematic_viscosity(
    kinematic_viscosity_of_air,
):
    D = 1
    T_s = 1
    T_a = 0
    Nu_f = kinematic_viscosity_of_air
    g = 273.65  # 0.5*T_s in Kelvin

    Gr = dimensionless.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(1 / (Nu_f**2))


@hypothesis.given(coefficient_of_gravity=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False))
def test_grashof_number_scales_linearly_with_gravity(coefficient_of_gravity):
    D = 1
    T_s = 1
    T_a = 0
    Nu_f = np.sqrt(1 / 273.65)
    g = coefficient_of_gravity

    Gr = dimensionless.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(g)


def test_grashof_number_with_example():
    D = 4
    T_s = 4
    T_a = 2
    Nu_f = 2
    g = 276.15

    Gr = dimensionless.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(32)


@hypothesis.given(thermal_conductivity=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False))
def test_prandtl_number_scales_inversely_with_thermal_conductivity(thermal_conductivity):
    c_f = 1
    mu_f = 1
    lambda_f = thermal_conductivity

    assert dimensionless.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(1 / lambda_f)


@hypothesis.given(dynamic_viscosity=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False))
def test_prandtl_number_scales_linearly_with_dynamic_viscosity(dynamic_viscosity):
    c_f = 1
    mu_f = dynamic_viscosity
    lambda_f = 1

    assert dimensionless.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(mu_f)


@hypothesis.given(
    specific_heat_capacity=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False)
)
def test_prandtl_number_scales_linearly_with_specific_heat_capacity(specific_heat_capacity):
    c_f = specific_heat_capacity
    mu_f = 1
    lambda_f = 1

    assert dimensionless.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(c_f)


def test_compute_prandtl_number_with_example():
    c_f = 0.5
    mu_f = 3
    lambda_f = 0.5

    assert dimensionless.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(3)
