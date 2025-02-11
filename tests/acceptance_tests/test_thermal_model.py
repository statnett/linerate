"""Test cases from Annex E of CIGRE TB 601."""

import numpy as np
import pytest

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
        ground_albedo=0.1,
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
        num_conductors=1,
    )


@pytest.fixture
def example_model_a(example_span_a, example_weather_a):
    return linerate.Cigre601(example_span_a, example_weather_a, np.datetime64("2016-06-10 11:00"))
