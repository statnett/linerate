"""Test cases to compare CIGRE TB 207 and IEEE738 to Ratekit."""

import numpy as np
import pytest
from pytest import approx

import linerate

MILES_PER_M = 1 / 1609.344
INCH_PER_M = 1 / 0.0254


@pytest.fixture
def parrot_conductor():
    return linerate.Conductor(
        core_diameter=0.1004 / INCH_PER_M,
        conductor_diameter=1.508 / INCH_PER_M,
        outer_layer_strand_diameter=0.1673 / INCH_PER_M,
        emissivity=0.8,
        solar_absorptivity=0.9,
        temperature1=25,
        temperature2=75,
        resistance_at_temperature2=0.0750 * MILES_PER_M,
        resistance_at_temperature1=0.0627 * MILES_PER_M,
        aluminium_cross_section_area=float("nan"),
        constant_magnetic_effect=1,
        current_density_proportional_magnetic_effect=0,
        max_magnetic_core_relative_resistance_increase=1,
    )


@pytest.fixture
def example_weather_a():
    return linerate.Weather(
        air_temperature=np.array(30),
        wind_direction=np.radians(0),
        wind_speed=0.60,
        ground_albedo=0.0,
        clearness_ratio=1,
    )


@pytest.fixture
def example_span_a(parrot_conductor):
    start_tower = linerate.Tower(latitude=65, longitude=0.0000, altitude=0)
    end_tower = linerate.Tower(latitude=65, longitude=0.0001, altitude=0)
    return linerate.Span(
        conductor=parrot_conductor,
        start_tower=start_tower,
        end_tower=end_tower,
        num_conductors=1,
    )


@pytest.fixture
def example_model_a(example_span_a, example_weather_a):
    return linerate.Cigre207(
        example_span_a,
        example_weather_a,
        np.datetime64("2016-06-21 12:00"),
        include_diffuse_radiation=False,
    )


def test_example_a_convective_cooling(example_model_a):
    assert example_model_a.compute_convective_cooling(50, None) == approx(32.52, abs=0.5)


def test_example_a_radiative_cooling(example_model_a):
    assert example_model_a.compute_radiative_cooling(50, None) == approx(13.41, abs=0.5)


def test_example_a_solar_heating(example_model_a):
    assert example_model_a.compute_solar_heating(50, None) == approx(31.04, abs=0.5)


def test_example_a_resistance(example_model_a):
    assert example_model_a.compute_resistance(50, None) == approx(0.0428 / 1e3, rel=1e-3)


def test_example_a_ampacity(example_model_a):
    assert example_model_a.compute_steady_state_ampacity(
        50, min_ampacity=0, max_ampacity=10000, tolerance=1e-8
    ) == approx(590, abs=1.5)


def test_example_a_convective_cooling_70(example_model_a):
    assert example_model_a.compute_convective_cooling(70, None) == approx(63.59, rel=3e-2)


def test_example_a_radiative_cooling_70(example_model_a):
    assert example_model_a.compute_radiative_cooling(70, None) == approx(29.56, abs=0.5)


def test_example_a_solar_heating_70(example_model_a):
    assert example_model_a.compute_solar_heating(70, None) == approx(31.05, abs=0.5)


def test_example_a_resistance_70(example_model_a):
    assert example_model_a.compute_resistance(70, None) == approx(0.0458 / 1e3, rel=1e-3)


def test_example_a_ampacity_70(example_model_a):
    assert example_model_a.compute_steady_state_ampacity(
        70, min_ampacity=0, max_ampacity=10000, tolerance=1e-8
    ) == approx(1178, abs=1.5)


@pytest.fixture
def example_model_b(example_span_a, example_weather_a):
    return linerate.IEEE738(example_span_a, example_weather_a, np.datetime64("2016-06-21 12:00"))


def test_example_b_solar_heating(example_model_b):
    assert example_model_b.compute_solar_heating(50, None) == approx(33.03, abs=0.5)
