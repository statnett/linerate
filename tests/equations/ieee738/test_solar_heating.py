import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx

import linerate.equations.ieee738.solar_heating as solar_heating
from linerate.equations.math import switch_cos_sin


@hypothesis.given(solar_altitude=st.floats(min_value=-10, max_value=10, allow_nan=False))
@pytest.mark.parametrize("industrial_atmosphere", [True, False])
def test_total_heat_flux_density_is_nonnegative(solar_altitude, industrial_atmosphere):
    H_c = solar_altitude
    sin_H_c = np.sin(H_c)

    Q_S = solar_heating.compute_total_heat_flux_density(sin_H_c, industrial_atmosphere)
    assert Q_S >= 0


@hypothesis.given(solar_altitude=st.floats(min_value=-10, max_value=10, allow_nan=False))
def test_total_heat_flux_density_clear_atmosphere_scales_correctly_with_solar_altitude(
    solar_altitude,
):
    sin_solar_altitude = np.sin(solar_altitude)

    H_c = np.degrees(np.arcsin(sin_solar_altitude))

    calc = (
        -42.2391
        + 63.8044 * H_c
        - 1.922 * H_c**2
        + 3.46921e-2 * H_c**3
        - 3.61118e-4 * H_c**4
        + 1.94318e-6 * H_c**5
        - 4.07608e-9 * H_c**6
    )
    Q_S = np.where(calc >= 0, calc, 0 * calc)
    estimated_Q_S = solar_heating.compute_total_heat_flux_density(sin_solar_altitude, True)
    assert Q_S == approx(estimated_Q_S)


@hypothesis.given(
    elevation_of_conductor_above_sea_level=st.floats(
        min_value=-10, max_value=10000, allow_nan=False
    )
)
def test_solar_altitude_correction_factor_is_nonnegative(elevation_of_conductor_above_sea_level):
    H_e = elevation_of_conductor_above_sea_level
    K_solar = solar_heating.compute_solar_altitude_correction_factor(H_e)
    assert K_solar >= 0


@hypothesis.given(
    height_above_sea_level_of_conductor=st.floats(min_value=-10, max_value=10000, allow_nan=False)
)
def test_solar_altitude_correction_factor_scales_correctly_with_height_above_sea_level_of_conductor(
    height_above_sea_level_of_conductor,
):
    H_e = height_above_sea_level_of_conductor

    K_solar = 1 + 1.148e-4 * H_e - 1.108e-8 * H_e**2
    assert K_solar == approx(solar_heating.compute_solar_altitude_correction_factor(H_e))


@hypothesis.given(absorptivity=st.floats(min_value=0.23, max_value=0.91, allow_nan=False))
def test_solar_heating_scales_linearly_with_absorptivity(absorptivity):
    alpha = absorptivity
    Q_se = 1
    cos_theta = 0
    A = 1
    q_s = solar_heating.compute_solar_heating(alpha, Q_se, cos_theta, A)
    assert q_s == approx(alpha)


@hypothesis.given(elevation_correction_factor=st.floats(allow_nan=False))
def test_solar_heating_scales_linearly_with_elevation_correction_factor(
    elevation_correction_factor,
):
    alpha = 1
    Q_se = elevation_correction_factor
    cos_theta = 0
    A = 1
    q_s = solar_heating.compute_solar_heating(alpha, Q_se, cos_theta, A)
    assert q_s == approx(elevation_correction_factor)


@hypothesis.given(
    solar_effective_incidence_angle=st.floats(min_value=0, max_value=np.pi, allow_nan=False)
)
def test_solar_heating_scales_linearly_with_cos_solar_effective_incidence_angle(
    solar_effective_incidence_angle,
):
    alpha = 1
    Q_se = 1
    cos_theta = np.cos(solar_effective_incidence_angle)
    sin_theta = switch_cos_sin(cos_theta)
    A = 1
    q_s = solar_heating.compute_solar_heating(alpha, Q_se, cos_theta, A)
    assert q_s == approx(sin_theta)


@hypothesis.given(
    projected_area_of_conductor=st.floats(min_value=0.001, max_value=1, allow_nan=False)
)
def test_solar_heating_scales_linearly_with_projected_area_of_conductor(
    projected_area_of_conductor,
):
    alpha = 1
    Q_se = 1
    cos_theta = 0
    A = projected_area_of_conductor
    q_s = solar_heating.compute_solar_heating(alpha, Q_se, cos_theta, A)
    assert q_s == approx(A)


def test_total_heat_flux_density_clear_atmosphere_with_example():
    sin_H_c = 0.7885372342
    Q_s = 974.5452942944816
    assert solar_heating.compute_total_heat_flux_density(sin_H_c, clear_atmosphere=True) == approx(
        Q_s
    )


def test_total_heat_flux_density_industrial_atmosphere_with_example():
    sin_H_c = np.sin(np.pi / 180)
    Q_s = 68.0233642086
    assert solar_heating.compute_total_heat_flux_density(sin_H_c, clear_atmosphere=False) == approx(
        Q_s
    )


def test_sin_solar_altitude_correction_factor_with_example():
    H_e = 250
    assert solar_heating.compute_solar_altitude_correction_factor(H_e) == approx(1.0280075)


def test_elevation_correction_factor_with_example():
    Q_s = 974.5452942944816
    K_solar = 1.0280075
    assert solar_heating.compute_elevation_correction_factor(K_solar, Q_s) == approx(
        1001.8398716244342
    )


def test_solar_heating_with_example():
    alpha = 0.8
    Q_se = 1001.8398716244342
    cos_theta = 0
    A = 0.04
    assert solar_heating.compute_solar_heating(alpha, Q_se, cos_theta, A) == approx(32.058875892)
