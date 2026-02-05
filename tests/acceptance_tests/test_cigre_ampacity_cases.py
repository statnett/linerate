"""Test cases from Annex E of CIGRE TB 601."""

import numpy as np
import pytest
from pytest import approx

from linerate.models.cigre601 import Cigre601
from linerate.models.thermal_model import ThermalModel
from linerate.types import Conductor, Span, Tower, Weather


def test_example_a_convective_cooling(example_model_1_conductors: ThermalModel):
    assert example_model_1_conductors.compute_convective_cooling(100) == approx(77.6, abs=0.5)


def test_example_a_radiative_cooling(example_model_1_conductors: ThermalModel):
    assert example_model_1_conductors.compute_radiative_cooling(100) == approx(39.1, abs=0.5)


def test_example_a_solar_heating(example_model_1_conductors: ThermalModel):
    assert example_model_1_conductors.compute_solar_heating() == approx(27.2, abs=0.5)


def test_example_a_resistance(example_model_1_conductors: ThermalModel):
    assert example_model_1_conductors.compute_resistance(100, np.nan) == approx(
        9.3905e-5, abs=0.0001e-5
    )


def test_example_a_ampacity(example_model_1_conductors: ThermalModel):
    # There are noticable roundoff errors in the report
    assert example_model_1_conductors.compute_steady_state_ampacity(100, tolerance=1e-8) == approx(
        976, abs=1.5
    )


@pytest.fixture
def drake_conductor_b() -> Conductor:
    return Conductor(
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
def example_weather_b() -> Weather:
    return Weather(
        air_temperature=20,
        wind_direction=np.radians(80),  # Conductor azimuth is 0, so angle of attack is 80
        wind_speed=1.66,
        ground_albedo=0.15,
        clearness_ratio=0.5,
    )


@pytest.fixture()
def example_span_b(drake_conductor_b: Conductor) -> Span:
    start_tower = Tower(latitude=50 - 0.0045, longitude=0, altitude=500 - 88)
    end_tower = Tower(latitude=50 + 0.0045, longitude=0, altitude=500 + 88)
    return Span(
        conductor=drake_conductor_b,
        start_tower=start_tower,
        end_tower=end_tower,
        num_conductors=1,
    )


def test_example_span_b_has_correct_altitude(example_span_b: Span):
    assert example_span_b.conductor_altitude == approx(500, abs=0.5)


def test_example_span_b_has_correct_inclination(example_span_b: Span):
    assert np.degrees(example_span_b.inclination) == approx(10, abs=0.5)


def test_example_span_b_has_correct_latitude(example_span_b: Span):
    assert example_span_b.latitude == approx(50)


@pytest.fixture()
def example_model_b(example_span_b: Span, example_weather_b: Weather) -> Cigre601:
    return Cigre601(example_span_b, example_weather_b, np.datetime64("2016-10-03 14:00"))


def test_example_b_convective_cooling(example_model_b: ThermalModel):
    assert example_model_b.compute_convective_cooling(100) == approx(172.1, abs=0.5)


def test_example_b_radiative_cooling(example_model_b: ThermalModel):
    assert example_model_b.compute_radiative_cooling(100) == approx(54, abs=0.5)


def test_example_b_solar_heating(example_model_b: ThermalModel):
    assert example_model_b.compute_solar_heating() == approx(13.7, abs=0.5)


def test_example_b_resistance(example_model_b: ThermalModel):
    assert example_model_b.compute_resistance(100, np.nan) == approx(9.3905e-5, abs=0.0001e-5)


def test_example_b_ampacity(example_model_b: ThermalModel):
    # There are noticable roundoff errors in the report
    # There is a typo in the report, where it says that the ampacity is 1054, but it is 1504.
    assert example_model_b.compute_steady_state_ampacity(100, tolerance=1e-8) == approx(
        1504, abs=1.5
    )
