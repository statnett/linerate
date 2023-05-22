"""Test cases from Annex E of CIGRE TB 601."""
import numpy as np
import pytest
from pytest import approx

import linerate


@pytest.fixture
def drake_conductor_a():
    return linerate.Conductor(
        core_diameter=10.4e-3,
        conductor_diameter=28.1e-3,
        outer_layer_strand_diameter=4.4e-3,
        emissivity=0.8,
        solar_absorptivity=0.8,
        temperature1=25,
        temperature2=75,
        resistance_at_temperature2=8.688e-5,
        resistance_at_temperature1=7.283e-5,
        aluminium_cross_section_area=float("nan"),
        constant_magnetic_effect=1,
        current_density_proportional_magnetic_effect=0,
        max_magnetic_core_relative_resistance_increase=1,
    )


@pytest.fixture
def example_weather_a():
    return linerate.Weather(
        air_temperature=40,
        wind_direction=np.radians(30),  # Conductor azimuth is 90, so 90 - 30 is 30
        wind_speed=0.61,
        clearness_ratio=1,
    )


@pytest.fixture
def example_span_a(drake_conductor_a):
    start_tower = linerate.Tower(latitude=30, longitude=0.0001, altitude=0)
    end_tower = linerate.Tower(latitude=30, longitude=-0.0001, altitude=0)
    return linerate.Span(
        conductor=drake_conductor_a,
        start_tower=start_tower,
        end_tower=end_tower,
        ground_albedo=0.1,
        num_conductors=1,
    )


@pytest.fixture
def example_model_a(example_span_a, example_weather_a):
    return linerate.Cigre601(example_span_a, example_weather_a, np.datetime64("2016-06-10 11:00"))


def test_example_a_convective_cooling(example_model_a):
    assert example_model_a.compute_convective_cooling(100, None) == approx(77.6, abs=0.5)


def test_example_a_radiative_cooling(example_model_a):
    assert example_model_a.compute_radiative_cooling(100, None) == approx(39.1, abs=0.5)


def test_example_a_solar_heating(example_model_a):
    assert example_model_a.compute_solar_heating(100, None) == approx(27.2, abs=0.5)


def test_example_a_resistance(example_model_a):
    assert example_model_a.compute_resistance(100, None) == approx(9.3905e-5, abs=0.0001e-5)


def test_example_a_ampacity(example_model_a):
    # There are noticable roundoff errors in the report
    assert example_model_a.compute_steady_state_ampacity(100, tolerance=1e-8) == approx(
        976, abs=1.5
    )


@pytest.fixture
def drake_conductor_b():
    return linerate.Conductor(
        core_diameter=10.4e-3,
        conductor_diameter=28.1e-3,
        outer_layer_strand_diameter=2.2e-3,
        emissivity=0.9,
        solar_absorptivity=0.9,
        temperature1=25,
        temperature2=75,
        resistance_at_temperature2=8.688e-5,
        resistance_at_temperature1=7.283e-5,
        aluminium_cross_section_area=float("nan"),
        constant_magnetic_effect=1,
        current_density_proportional_magnetic_effect=0,
        max_magnetic_core_relative_resistance_increase=1,
    )


@pytest.fixture
def example_weather_b():
    return linerate.Weather(
        air_temperature=20,
        wind_direction=np.radians(80),  # Conductor azimuth is 0, so angle of attack is 80
        wind_speed=1.66,
        clearness_ratio=0.5,
    )


@pytest.fixture()
def example_span_b(drake_conductor_b):
    start_tower = linerate.Tower(latitude=50 - 0.0045, longitude=0, altitude=500 - 88)
    end_tower = linerate.Tower(latitude=50 + 0.0045, longitude=0, altitude=500 + 88)
    return linerate.Span(
        conductor=drake_conductor_b,
        start_tower=start_tower,
        end_tower=end_tower,
        ground_albedo=0.15,
        num_conductors=1,
    )


def test_example_span_b_has_correct_altitude(example_span_b):
    assert example_span_b.conductor_altitude == approx(500, abs=0.5)


def test_example_span_b_has_correct_inclination(example_span_b):
    assert np.degrees(example_span_b.inclination) == approx(10, abs=0.5)


def test_example_span_b_has_correct_latitude(example_span_b):
    assert example_span_b.latitude == approx(50)


@pytest.fixture()
def example_model_b(example_span_b, example_weather_b):
    return linerate.Cigre601(example_span_b, example_weather_b, np.datetime64("2016-10-03 14:00"))


def test_example_b_convective_cooling(example_model_b):
    assert example_model_b.compute_convective_cooling(100, None) == approx(172.1, abs=0.5)


def test_example_b_radiative_cooling(example_model_b):
    assert example_model_b.compute_radiative_cooling(100, None) == approx(54, abs=0.5)


def test_example_b_solar_heating(example_model_b):
    assert example_model_b.compute_solar_heating(100, None) == approx(13.7, abs=0.5)


def test_example_b_resistance(example_model_b):
    assert example_model_b.compute_resistance(100, None) == approx(9.3905e-5, abs=0.0001e-5)


def test_example_b_ampacity(example_model_b):
    # There are noticable roundoff errors in the report
    # There is a typo in the report, where it says that the ampacity is 1054, but it is 1504.
    assert example_model_b.compute_steady_state_ampacity(100, tolerance=1e-8) == approx(
        1504, abs=1.5
    )
