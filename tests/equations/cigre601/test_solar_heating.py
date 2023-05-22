import hypothesis
import hypothesis.strategies as st
import numpy as np
from pytest import approx

import linerate.equations.cigre601.solar_heating as solar_heating


@hypothesis.given(
    solar_altitude=st.floats(),
    clearness_ratio=st.floats(
        allow_nan=False, min_value=0, max_value=3
    ),  # Recommended values between 0 and 1.4
    height_above_sea_level=st.floats(
        allow_nan=False, min_value=0, max_value=10_000
    ),  # Tallest mountain on earth is lower than 10 000 m
)
def test_direct_solar_radiation_nonnegative(
    solar_altitude, clearness_ratio, height_above_sea_level
):
    N_s = clearness_ratio
    y = height_above_sea_level
    sin_H_s = np.sin(solar_altitude)

    direct_solar_radiation = solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y)

    if any(np.isnan([N_s, y, sin_H_s])):
        assert np.isnan(direct_solar_radiation)
    else:
        assert direct_solar_radiation >= 0


@hypothesis.given(
    clearness_ratio=st.floats(
        allow_nan=False, min_value=0, max_value=10
    ),  # Recommended values between 0 and 1.4
)
def test_direct_solar_radiation_scales_linearly_with_clearness_ratio(clearness_ratio):
    y = 0
    N_s = clearness_ratio
    sin_H_s = 1

    assert solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y) == approx(
        N_s * 1280 / 1.314
    )


@hypothesis.given(
    height_above_sea_level=st.floats(
        allow_nan=False, min_value=0, max_value=10_000
    ),  # Tallest mountain on earth is lower than 10 000 m
)
def test_direct_solar_radiation_scales_affinely_with_height_above_sea_level(height_above_sea_level):
    y = height_above_sea_level
    N_s = 1
    sin_H_s = 1

    I_B_0 = 1280 / 1.314
    assert solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y) / I_B_0 - 1 == approx(
        y * 1.4e-4 * (1367 / I_B_0 - 1), rel=1e-8
    )


@hypothesis.given(
    solar_altitude=st.floats(allow_nan=False, allow_infinity=False),
)
def test_direct_solar_radiation_scales_correctly_with_sin_solar_altitude(solar_altitude):
    y = 0
    N_s = 1
    sin_H_s = np.sin(solar_altitude)
    I_B = solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y)

    if sin_H_s > 0:
        assert I_B / 1280 == approx(sin_H_s / (sin_H_s + 0.314))
    else:
        assert I_B == 0


def test_direct_solar_radiation_with_example():
    y = 0
    N_s = 0.5
    sin_H_s = 0.314  # sin(H_s) / (sin(H_s) + 0.314) = 0.5

    I_B_0 = 320
    assert solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y) == approx(I_B_0)

    y_intercept = 4.5806245e-4  # = 1.4e-4 * (1367/I_B_0 - 1)
    y = 0.5 / y_intercept
    I_B = 1.5 * 320

    assert solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y) == approx(I_B)


@hypothesis.given(
    solar_altitude=st.floats(),
    direct_solar_radiation=st.floats(
        allow_nan=False, min_value=0, max_value=5000
    ),  # Cannot be greater than 5000 (approximately 4x solar constant)
)
def test_diffuse_sky_radiation_radiation_nonnegative(solar_altitude, direct_solar_radiation):
    I_B = direct_solar_radiation
    sin_H_s = np.sin(solar_altitude)

    I_d = solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)

    if any(np.isnan([I_B, sin_H_s])):
        assert np.isnan(I_d)
    else:
        assert I_d >= 0


@hypothesis.given(
    solar_altitude=st.floats(allow_nan=False, allow_infinity=False),
)
def test_diffuse_sky_radiation_scales_linearly_with_sin_solar_altitude(solar_altitude):
    sin_H_s = np.sin(solar_altitude)
    I_B = 0
    I_d = solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)

    if sin_H_s > 0:
        assert I_d == approx(430.5 * sin_H_s, rel=1e-8)
    else:
        assert I_d == 0


@hypothesis.given(
    direct_solar_radiation=st.floats(allow_nan=False, allow_infinity=False),
)
def test_diffuse_sky_radiation_scales_affinely_with_direct_solar_radiation(direct_solar_radiation):
    sin_H_s = 1
    I_B = direct_solar_radiation
    I_d = solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)

    if I_d > 0:
        assert (I_d - 430.5) == approx(-0.3288 * I_B, rel=1e-8)
    else:
        assert 430.5 - 0.3288 * I_B < 0


def test_diffuse_sky_radiation_with_example():
    sin_H_s = 0.25
    I_B = 0.5 / 0.3288
    I_d = solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)

    assert I_d == approx(430 / 4, rel=1e-8)


@hypothesis.given(diffuse_sky_radiation=st.floats(allow_nan=False))
def test_global_radiation_intensity_scales_affinely_with_diffuse_sky_radiation(
    diffuse_sky_radiation,
):
    I_B = 1
    sin_eta = -1
    F = 2 / np.pi
    I_d = diffuse_sky_radiation
    sin_H_s = 1

    I_T = solar_heating.compute_global_radiation_intensity(I_B, I_d, F, sin_eta, sin_H_s)
    assert I_T == approx(2 * I_d)


@hypothesis.given(direct_solar_radiation=st.floats(allow_nan=False))
def test_global_radiation_intensity_scales_affinely_with_direct_radiation(direct_solar_radiation):
    I_B = direct_solar_radiation
    sin_eta = 1
    F = 2 / np.pi
    I_d = 1
    sin_H_s = 1

    I_T = solar_heating.compute_global_radiation_intensity(I_B, I_d, F, sin_eta, sin_H_s)
    assert I_T == approx(2 * I_B + 2)


@hypothesis.given(albedo=st.floats(allow_nan=False))
def test_global_radiation_intensity_scales_affinely_with_albedo(albedo):
    I_B = 1
    sin_eta = 1
    F = albedo
    I_d = 1
    sin_H_s = 1

    I_T = solar_heating.compute_global_radiation_intensity(I_B, I_d, F, sin_eta, sin_H_s)
    assert I_T == approx(np.pi * F + 2)


@hypothesis.given(sin_solar_altitude=st.floats(allow_nan=False))
def test_global_radiation_intensity_scales_affinely_sin_solar_altitude(sin_solar_altitude):
    I_B = 1
    sin_eta = 1
    F = 2 / np.pi
    I_d = 1
    sin_H_s = sin_solar_altitude

    I_T = solar_heating.compute_global_radiation_intensity(I_B, I_d, F, sin_eta, sin_H_s)
    assert I_T == approx(sin_H_s + 3)


@hypothesis.given(sin_angle_of_sun_on_line=st.floats(allow_nan=False))
def test_global_radiation_intensity_scales_affinely_sin_angle_of_sun_on_line(
    sin_angle_of_sun_on_line,
):
    I_B = 1
    sin_eta = sin_angle_of_sun_on_line
    F = 2 / np.pi
    I_d = 1
    sin_H_s = 1

    I_T = solar_heating.compute_global_radiation_intensity(I_B, I_d, F, sin_eta, sin_H_s)
    assert I_T == approx(sin_eta + 3)


def test_global_radiation_intensity_with_examples():
    I_B = 1 / 0.75
    sin_eta = 0.5
    sin_H_s = 0.5
    F = 1 / np.pi
    I_d = 1

    I_T = solar_heating.compute_global_radiation_intensity(I_B, I_d, F, sin_eta, sin_H_s)
    assert I_T == approx(2.5)


@hypothesis.given(conductor_diameter=st.floats(allow_nan=False))
def test_solar_heating_scales_linearly_with_conductor_diameter(conductor_diameter):
    D = conductor_diameter
    alpha_s = 1
    I_T = 1

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(D)


@hypothesis.given(solar_absorptivity=st.floats(allow_nan=False))
def test_solar_heating_scales_linearly_with_solar_absorptivity(solar_absorptivity):
    D = 1
    alpha_s = solar_absorptivity
    I_T = 1

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(alpha_s)


@hypothesis.given(global_radiation_intensity=st.floats(allow_nan=False))
def test_solar_heating_scales_linearly_with_global_radiation_intensity(global_radiation_intensity):
    D = 1
    alpha_s = 1
    I_T = global_radiation_intensity

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(
        global_radiation_intensity
    )


def test_solar_heating_scales_linearly_with_example():
    D = 0.6
    alpha_s = 0.5
    I_T = 2

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(0.6)
