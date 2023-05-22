import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx
from scipy.interpolate import lagrange

import linerate.equations.ieee738.convective_cooling as convective_cooling


@hypothesis.given(
    temperature_of_conductor_surface=st.floats(min_value=-273, max_value=1000, allow_nan=False)
)
def test_air_temperature_at_boundary_level_scales_with_temperature_of_conductor_surface(
    temperature_of_conductor_surface,
):
    T_s = temperature_of_conductor_surface
    T_a = 0

    assert convective_cooling.compute_air_temperature_at_boundary_layer(T_s, T_a) == approx(T_s / 2)


@hypothesis.given(
    temperature_of_ambient_air=st.floats(min_value=-273, max_value=1000, allow_nan=False)
)
def test_air_temperature_at_boundary_level_scales_with_temperature_of_ambient_air(
    temperature_of_ambient_air,
):
    T_s = 0
    T_a = temperature_of_ambient_air

    assert convective_cooling.compute_air_temperature_at_boundary_layer(T_s, T_a) == approx(T_a / 2)


@hypothesis.given(
    temperature_of_conductor_surface=st.floats(min_value=-273, max_value=1000, allow_nan=False),
    temperature_of_ambient_air=st.floats(min_value=-273, max_value=1000, allow_nan=False),
)
def test_air_temperature_at_boundary_level_scales_with_ranges_of_temperature(
    temperature_of_conductor_surface, temperature_of_ambient_air
):
    T_s = temperature_of_conductor_surface
    T_a = temperature_of_ambient_air

    assert convective_cooling.compute_air_temperature_at_boundary_layer(T_s, T_a) == approx(
        (T_a + T_s) / 2
    )


def test_air_temperature_at_boundary_level_with_example():
    T_s = 1
    T_a = 1

    assert convective_cooling.compute_air_temperature_at_boundary_layer(T_s, T_a) == approx(1)


@hypothesis.given(
    air_temperature_at_boundary_layer=st.floats(min_value=-273, max_value=1000, allow_nan=False)
)
def test_dynamic_viscosity_of_air_scales_with_air_temperature_at_boundary_layer(
    air_temperature_at_boundary_layer,
):
    T_film = air_temperature_at_boundary_layer

    calc = (1.458e-6 * (T_film + 273) ** 1.5) / (T_film + 383.4)

    assert convective_cooling.compute_dynamic_viscosity_of_air(T_film) == approx(calc)


def test_dynamic_viscosity_of_air_with_example():
    T_film_1 = -273
    T_film_2 = 0

    assert convective_cooling.compute_dynamic_viscosity_of_air(T_film_1) == approx(0)
    assert convective_cooling.compute_dynamic_viscosity_of_air(T_film_2) == approx(
        0.00001715336725523064
    )


@hypothesis.given(dynamic_viscosity_of_air=st.floats(allow_nan=False, allow_infinity=False))
def test_kinematic_viscosity_of_air_scales_linearly_with_dynamic_viscosity(
    dynamic_viscosity_of_air,
):
    mu_f = dynamic_viscosity_of_air
    rho_f = 1

    assert convective_cooling.compute_kinematic_viscosity_of_air(mu_f, rho_f) == approx(mu_f)


@hypothesis.given(air_density=st.floats(min_value=1e-8, allow_nan=False, allow_infinity=False))
def test_kinematic_viscosity_of_air_scales_inversely_with_density(air_density):
    mu_f = 1
    rho_f = air_density

    assert convective_cooling.compute_kinematic_viscosity_of_air(mu_f, rho_f) == approx(1 / rho_f)


def test_kinematic_viscosity_of_air_with_example():
    mu_f = 4
    rho_f = 4

    assert convective_cooling.compute_kinematic_viscosity_of_air(mu_f, rho_f) == approx(1)


@hypothesis.given(wind_speed=st.floats(min_value=0, max_value=1000, allow_nan=False))
def test_reynolds_number_scales_linearly_with_wind_speed(wind_speed):
    v = wind_speed
    D = 1
    nu_f = 1

    assert convective_cooling.compute_reynolds_number(v, D, nu_f) == pytest.approx(v)


@hypothesis.given(conductor_diameter=st.floats(min_value=0, max_value=1000, allow_nan=False))
def test_reynolds_number_scales_linearly_with_diameter(conductor_diameter):
    v = 1
    D = conductor_diameter
    nu_f = 1

    assert convective_cooling.compute_reynolds_number(v, D, nu_f) == pytest.approx(D)


@hypothesis.given(kinematic_viscosity=st.floats(min_value=1e-10, allow_nan=False))
def test_reynolds_number_scales_inversly_with_kinematic_viscosity(kinematic_viscosity):
    v = 1
    D = 1
    nu_f = kinematic_viscosity

    assert convective_cooling.compute_reynolds_number(v, D, nu_f) == approx(1 / nu_f)


def test_reynolds_number_with_example():
    v = 0.1
    D = 1.2
    nu_f = 10

    assert convective_cooling.compute_reynolds_number(v, D, nu_f) == approx(0.012)


def test_wind_direction_factor_with_example():
    Phi = 0
    assert convective_cooling.compute_wind_direction_factor(Phi) == approx(0.388)


@hypothesis.given(
    air_temperature_at_boundary_layer=st.floats(min_value=0, max_value=1000, allow_nan=False)
)
def test_thermal_conductivity_of_air_scales_with_air_temperature_at_boundary_layer(
    air_temperature_at_boundary_layer,
):
    T_film = air_temperature_at_boundary_layer

    k_f = 2.424e-2 + 7.477e-5 * T_film - 4.407e-9 * T_film**2
    assert convective_cooling.compute_thermal_conductivity_of_air(T_film) == approx(k_f)


@hypothesis.given(
    air_temperature_at_boundary_layer=st.floats(min_value=0, max_value=1000, allow_nan=False)
)
def test_thermal_conductivity_of_air_has_correct_interpolant(air_temperature_at_boundary_layer):
    T_film = np.arange(3) + air_temperature_at_boundary_layer
    _k_f = convective_cooling.compute_thermal_conductivity_of_air(T_film)
    lagrange_poly = lagrange(T_film, _k_f)

    a = 2.424e-2
    b = 7.477e-5
    c = -4.407e-9

    np.testing.assert_allclose(lagrange_poly.coef, [c, b, a])


def test_thermal_conductivity_of_air_with_example():
    T_film_1 = 10
    T_film_2 = 23.5
    assert convective_cooling.compute_thermal_conductivity_of_air(T_film_1) == approx(0.0249872593)
    assert convective_cooling.compute_thermal_conductivity_of_air(T_film_2) == approx(0.02599466123)


def test_forced_convection_chooses_largest_value():
    K_angle = 1
    N_Re_1 = 100
    N_Re_2 = 5000
    k_f = 1
    T_s = 10
    T_a = 9

    assert convective_cooling.compute_forced_convection(K_angle, N_Re_1, k_f, T_s, T_a) == approx(
        15.8124556479
    )
    assert convective_cooling.compute_forced_convection(K_angle, N_Re_2, k_f, T_s, T_a) == approx(
        124.954916454
    )


@hypothesis.given(
    air_temperature_at_boundary_layer=st.floats(min_value=0, max_value=1000, allow_nan=False)
)
def test_air_density_scales_with_air_temperature_at_boundary_layer(
    air_temperature_at_boundary_layer,
):
    H_e = 0
    T_film = air_temperature_at_boundary_layer

    assert convective_cooling.compute_air_density(T_film, H_e) == approx(
        1.293 / (1 + 0.00367 * T_film)
    )


@hypothesis.given(elevation=st.floats(min_value=0, max_value=10000, allow_nan=False))
def test_air_density_scales_with_elevation(elevation):
    H_e = elevation
    T_film = 0

    a = 1.293
    b = -1.525e-4
    c = 6.379e-9

    assert convective_cooling.compute_air_density(T_film, H_e) == approx(a + b * H_e + c * H_e**2)


@hypothesis.given(elevation=st.floats(min_value=0, max_value=10000, allow_nan=False))
def test_air_density_has_correct_interpolant(elevation):
    H_e = np.arange(3) + elevation
    T_film = 0
    _rho_f = convective_cooling.compute_air_density(T_film, H_e)
    lagrange_poly = lagrange(H_e, _rho_f)

    a = 1.293
    b = -1.525e-4
    c = 6.379e-9

    np.testing.assert_allclose(lagrange_poly.coef, [c, b, a])


def test_air_density_with_example():
    H_e = 0
    T_film = 0

    assert convective_cooling.compute_air_density(H_e, T_film) == approx(1.293)


@hypothesis.given(air_density=st.floats(min_value=1e-10, max_value=1000, allow_nan=False))
def test_natural_convection_scales_with_air_density(air_density):
    rho_f = air_density
    D_0 = 1 / (3.645 ** (4 / 3))
    T_s = 10
    T_a = 9

    assert convective_cooling.compute_natural_convection(rho_f, D_0, T_s, T_a) == approx(
        rho_f**0.5
    )


@hypothesis.given(diameter_of_conductor=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False))
def test_natural_convection_scales_with_diameter_of_conductor(diameter_of_conductor):
    rho_f = 1 / (3.645**2)
    D_0 = diameter_of_conductor
    T_s = 10
    T_a = 9

    assert convective_cooling.compute_natural_convection(rho_f, D_0, T_s, T_a) == approx(
        D_0**0.75
    )


@hypothesis.given(
    temperature_of_conductor_surface=st.floats(min_value=-273, max_value=1000, allow_nan=False)
)
def test_natural_convection_scales_with_temperature_of_conductor_surface(
    temperature_of_conductor_surface,
):
    rho_f = 1 / (3.645**2)
    D_0 = 1
    T_s = temperature_of_conductor_surface
    T_a = -20 if -20 <= T_s else T_s

    assert convective_cooling.compute_natural_convection(rho_f, D_0, T_s, T_a) == approx(
        (T_s - T_a) ** 1.25
    )


@hypothesis.given(
    temperature_of_ambient_air=st.floats(min_value=-273, max_value=1000, allow_nan=False)
)
def test_natural_convection_scales_with_temperature_of_ambient_air(temperature_of_ambient_air):
    rho_f = 1 / (3.645**2)
    D_0 = 1
    T_a = temperature_of_ambient_air

    T_s = 0 if 0 >= T_a else T_a

    assert convective_cooling.compute_natural_convection(rho_f, D_0, T_s, T_a) == approx(
        (T_s - T_a) ** 1.25
    )


@hypothesis.given(
    temperature_of_conductor_surface=st.floats(min_value=-273, max_value=1000, allow_nan=False),
    temperature_of_ambient_air=st.floats(min_value=-273, max_value=1000, allow_nan=False),
)
def test_natural_convection_with_ranges_of_temperature(
    temperature_of_conductor_surface, temperature_of_ambient_air
):
    rho_f = 1 / (3.645**2)
    D_0 = 1
    T_s = temperature_of_conductor_surface
    T_a = temperature_of_ambient_air

    assert convective_cooling.compute_natural_convection(rho_f, D_0, T_s, T_a) == approx(
        (T_s - T_a) ** 1.25
    )


def test_natural_convection_with_example():
    rho_f = 4
    D_0 = 5
    T_s = 40
    T_a = 20

    assert convective_cooling.compute_natural_convection(rho_f, D_0, T_s, T_a) == approx(
        1030.96168697
    )
