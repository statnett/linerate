import numpy as np
import pytest
from pytest import approx
from linerate import Conductor, Weather, Tower, Span, Cigre601


@pytest.fixture
def drake_conductor_transient() -> Conductor:
    return Conductor(
        conductor_diameter=28.143e-3,
        core_diameter=10.4e-3,
        outer_layer_strand_diameter=4.4e-3,
        emissivity=0.8,
        solar_absorptivity=0.8,
        temperature1=25,
        temperature2=75,
        resistance_at_temperature1=0.0727e-3,
        resistance_at_temperature2=0.0872e-3,
        aluminium_cross_section_area=float("nan"),
        constant_magnetic_effect=1,
        current_density_proportional_magnetic_effect=0,
        max_magnetic_core_relative_resistance_increase=1,
        steel_mass_per_unit_length=0.5119,
        steel_specific_heat_capacity_at_20_celsius=481,
        steel_specific_heat_capacity_temperature_coefficient=1e-4,
        aluminum_mass_per_unit_length=1.116,
        aluminum_specific_heat_capacity_at_20_celsius=897,
        aluminum_specific_heat_capacity_temperature_coefficient=3.8e-4,
    )


@pytest.fixture
def initial_weather() -> Weather:
    return Weather(
        air_temperature=24,
        wind_speed=1.9,
        wind_direction=np.radians(35),
        ground_albedo=1.0,
    )


@pytest.fixture
def weather_1() -> Weather:
    return Weather(
        air_temperature=23.7,
        wind_speed=1.7,
        wind_direction=np.radians(28),
        ground_albedo=1.0,
    )


@pytest.fixture
def weather_2() -> Weather:
    return Weather(
        air_temperature=23.5,
        wind_speed=0.8,
        wind_direction=np.radians(53),
        ground_albedo=1.0,
    )


@pytest.fixture
def example_span_transient(drake_conductor_transient: Conductor) -> Span:
    start_tower = Tower(latitude=50, longitude=0.0001, altitude=0)
    end_tower = Tower(latitude=50, longitude=-0.0001, altitude=0)
    return Span(
        conductor=drake_conductor_transient,
        start_tower=start_tower,
        end_tower=end_tower,
        num_conductors=1,
    )


@pytest.fixture
def transient_initial_model(example_span_transient, initial_weather) -> Cigre601:
    return Cigre601(
        span=example_span_transient, weather=initial_weather, time=np.datetime64("2016-06-10 00:00")
    )


@pytest.fixture
def transient_model_1(example_span_transient, weather_1) -> Cigre601:
    return Cigre601(
        span=example_span_transient, weather=weather_1, time=np.datetime64("2016-06-10 00:10")
    )


@pytest.fixture
def transient_model_2(example_span_transient, weather_2) -> Cigre601:
    return Cigre601(
        span=example_span_transient, weather=weather_2, time=np.datetime64("2016-06-10 00:20")
    )


def test_initial_condition(transient_initial_model):
    assert transient_initial_model.compute_conductor_temperature(802, tolerance=0.01) == approx(
        42.01, abs=0.01
    )


def test_tracking(transient_initial_model, transient_model_1, transient_model_2):
    start_temperature = transient_initial_model.compute_conductor_temperature(802, tolerance=0.01)
    results_1 = [
        transient_model_1.compute_final_temperature(start_temperature, np.timedelta64(t, "m"), 819)
        for t in range(1, 11)
    ]
    results_2 = [
        transient_model_2.compute_final_temperature(results_1[-1], np.timedelta64(t, "m"), 856)
        for t in range(1, 11)
    ]
    correct_1 = [42.175, 42.321, 42.449, 42.562, 42.662, 42.750, 42.828, 42.897, 42.958, 43.011]
    correct_2 = [44.147, 45.199, 46.174, 47.075, 47.910, 48.682, 49.396, 50.057, 50.668, 51.233]
    assert all(result == approx(correct, abs=0.01) for result, correct in zip(results_1, correct_1))
    assert all(result == approx(correct, abs=0.01) for result, correct in zip(results_2, correct_2))
