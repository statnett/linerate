"""Test cases from Annex B and C of IEEE738.
"""
import numpy as np
import pytest
from pytest import approx

import linerate


@pytest.fixture
def drake_conductor_B():
    return linerate.Conductor(
        conductor_diameter=0.02812,
        emissivity=0.5,
        solar_absorptivity=0.5,
        temperature1=25,
        temperature2=75,
        resistance_at_temperature1=7.284e-5,
        resistance_at_temperature2=8.689e-5,
        core_diameter=10.4e-3,
        outer_layer_strand_diameter=4.4e-3,
        aluminium_cross_section_area=float("nan"),
        constant_magnetic_effect=1,
        current_density_proportional_magnetic_effect=0,
        max_magnetic_core_relative_resistance_increase=1,
    )


@pytest.fixture
def example_weather_B():
    return linerate.Weather(
        air_temperature=40,
        wind_direction=np.radians(90),
        wind_speed=0.61,
        clearness_ratio=1,
    )


@pytest.fixture
def example_span_B(drake_conductor_B):
    start_tower = linerate.Tower(latitude=43, longitude=0, altitude=0)
    end_tower = linerate.Tower(latitude=43, longitude=0, altitude=0)
    return linerate.Span(
        conductor=drake_conductor_B,
        start_tower=start_tower,
        end_tower=end_tower,
        ground_albedo=0.5,
        num_conductors=1,
    )


@pytest.fixture
def example_model_B(example_span_B, example_weather_B):
    return linerate.IEEE738(example_span_B, example_weather_B, np.datetime64("2016-06-10 14:00"))


def test_example_B_solar_heating(example_model_B):
    assert example_model_B.compute_solar_heating(100.7, 1000) == approx(13.738, abs=0.5)


def test_example_B_convective_cooling(example_model_B):
    assert example_model_B.compute_convective_cooling(100.7, 1000) == approx(83.061, abs=0.5)


@pytest.fixture
def example_span_C(drake_conductor_B):
    start_tower = linerate.Tower(latitude=43, longitude=0, altitude=0)
    end_tower = linerate.Tower(latitude=43, longitude=0, altitude=0)
    return linerate.Span(
        conductor=drake_conductor_B,
        start_tower=start_tower,
        end_tower=end_tower,
        ground_albedo=0.5,
        num_conductors=2,
    )


@pytest.fixture
def example_model_C(example_span_C, example_weather_B):
    return linerate.IEEE738(example_span_C, example_weather_B, np.datetime64("2016-06-10 14:00"))


def test_example_C_solar_heating(example_model_C):
    assert example_model_C.compute_solar_heating(101.1, 1003) == approx(13.732, abs=0.5)


def test_example_C_convective_cooling(example_model_C):
    assert example_model_C.compute_convective_cooling(101.1, 1003) == approx(83.600, abs=0.5)
