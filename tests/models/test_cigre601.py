import dataclasses

import hypothesis
import numpy as np
from hypothesis import HealthCheck

from linerate.equations import cigre601, solar_angles
from linerate.models.cigre601 import Cigre601, Cigre601WithSolarRadiation
from linerate.types import Span, Weather, WeatherWithSolarRadiation
from linerate.units import Date
from tests.conftest import numpy_datetimes


def get_equivalent_model_with_solar_radiation(
    model: Cigre601,
) -> Cigre601WithSolarRadiation:
    sin_H_s = solar_angles.compute_sin_solar_altitude_for_span(model.span, model.time)
    y = model.span.conductor_altitude
    N_s = model.weather.clearness_ratio

    I_B = cigre601.solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y)
    I_d = cigre601.solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)

    weather = WeatherWithSolarRadiation(
        air_temperature=model.weather.air_temperature,
        wind_direction=model.weather.wind_direction,
        wind_speed=model.weather.wind_speed,
        ground_albedo=model.weather.ground_albedo,
        direct_radiation_intensity=I_B,
        diffuse_radiation_intensity=I_d,
    )

    return Cigre601WithSolarRadiation(model.span, weather, model.time)


def test_solar_heating_with_solar_radiation_equals_with_cigre601_radiation(
    example_model_1_conductors: Cigre601,
):
    model_with_radiation = get_equivalent_model_with_solar_radiation(example_model_1_conductors)

    np.testing.assert_allclose(
        model_with_radiation.compute_solar_heating(),
        example_model_1_conductors.compute_solar_heating(),
    )


def test_ampacity_with_solar_radiation_equals_with_cigre601_radiation(
    example_model_1_conductors: Cigre601,
):
    model_with_radiation = get_equivalent_model_with_solar_radiation(example_model_1_conductors)

    np.testing.assert_allclose(
        model_with_radiation.compute_steady_state_ampacity(90.0),
        example_model_1_conductors.compute_steady_state_ampacity(90.0),
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_compute_global_radiation_intensity_increases_with_albedo(
    example_span_1_conductor: Span,
    example_weather_a: Weather,
    when: Date,
):
    weather_low_albedo = dataclasses.replace(example_weather_a, ground_albedo=0.0)
    weather_high_albedo = dataclasses.replace(example_weather_a, ground_albedo=0.9)

    model_low = Cigre601(example_span_1_conductor, weather_low_albedo, when)
    model_high = Cigre601(example_span_1_conductor, weather_high_albedo, when)

    assert (
        model_low.compute_global_radiation_intensity()
        <= model_high.compute_global_radiation_intensity()
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_compute_global_radiation_intensity_increases_with_clearness_ratio(
    example_span_1_conductor: Span,
    example_weather_a: Weather,
    when: Date,
):
    weather_low_clear = dataclasses.replace(example_weather_a, clearness_ratio=0.2)
    weather_high_clear = dataclasses.replace(example_weather_a, clearness_ratio=1.0)

    model_low = Cigre601(example_span_1_conductor, weather_low_clear, when)
    model_high = Cigre601(example_span_1_conductor, weather_high_clear, when)

    assert (
        model_low.compute_global_radiation_intensity()
        <= model_high.compute_global_radiation_intensity()
    )


def test_compute_global_radiation_intensity_example(
    example_span_1_conductor: Span,
    example_weather_a: Weather,
):
    when = np.datetime64("2022-06-01T12:00")
    model = Cigre601(example_span_1_conductor, example_weather_a, when)

    assert model.compute_global_radiation_intensity() == 1250.0362863902146


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_cigre601_with_solar_radiation_compute_global_radiation_intensity_increases_with_albedo(
    example_span_1_conductor: Span,
    example_weather_with_rad: WeatherWithSolarRadiation,
    when: Date,
):
    weather_low_albedo = dataclasses.replace(example_weather_with_rad, ground_albedo=0.0)
    weather_high_albedo = dataclasses.replace(example_weather_with_rad, ground_albedo=0.9)

    model_low = Cigre601WithSolarRadiation(example_span_1_conductor, weather_low_albedo, when)
    model_high = Cigre601WithSolarRadiation(example_span_1_conductor, weather_high_albedo, when)

    assert (
        model_low.compute_global_radiation_intensity()
        <= model_high.compute_global_radiation_intensity()
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_cigre601_with_solar_radiation_compute_global_radiation_intensity_increases_with_direct_radiation(
    example_span_1_conductor: Span,
    example_weather_with_rad: WeatherWithSolarRadiation,
    when: Date,
):
    weather_low_direct = dataclasses.replace(
        example_weather_with_rad,
        direct_radiation_intensity=300,
    )
    weather_high_direct = dataclasses.replace(
        example_weather_with_rad,
        direct_radiation_intensity=600,
    )

    model_low = Cigre601WithSolarRadiation(example_span_1_conductor, weather_low_direct, when)
    model_high = Cigre601WithSolarRadiation(example_span_1_conductor, weather_high_direct, when)

    assert (
        model_low.compute_global_radiation_intensity()
        <= model_high.compute_global_radiation_intensity()
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_cigre601_with_solar_radiation_compute_global_radiation_intensity_increases_with_diffuse_radiation(
    example_span_1_conductor: Span,
    example_weather_with_rad: WeatherWithSolarRadiation,
    when: Date,
):
    weather_low_direct = dataclasses.replace(
        example_weather_with_rad,
        diffuse_radiation_intensity=300,
    )
    weather_high_direct = dataclasses.replace(
        example_weather_with_rad,
        diffuse_radiation_intensity=600,
    )

    model_low = Cigre601WithSolarRadiation(example_span_1_conductor, weather_low_direct, when)
    model_high = Cigre601WithSolarRadiation(example_span_1_conductor, weather_high_direct, when)

    assert (
        model_low.compute_global_radiation_intensity()
        <= model_high.compute_global_radiation_intensity()
    )


def test_cigre601_with_solar_radiation_compute_global_radiation_intensity_example(
    example_span_1_conductor: Span,
    example_weather_with_rad: WeatherWithSolarRadiation,
):
    when = np.datetime64("2022-06-01T12:00")

    model = Cigre601WithSolarRadiation(example_span_1_conductor, example_weather_with_rad, when)

    assert model.compute_global_radiation_intensity() == 1040.4311074904813
