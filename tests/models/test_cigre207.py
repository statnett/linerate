import dataclasses

import hypothesis
import numpy as np
from hypothesis import HealthCheck
from pytest import approx

from linerate.models.cigre207 import Cigre207
from linerate.types import Span, Weather
from linerate.units import Date
from tests.conftest import numpy_datetimes


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_compute_global_radiation_intensity_ignores_albedo_without_diffuse_radiation(
    example_span_1_conductor: Span,
    example_weather_a: Weather,
    when: Date,
):
    weather_low_albedo = dataclasses.replace(example_weather_a, ground_albedo=0.0)
    weather_high_albedo = dataclasses.replace(example_weather_a, ground_albedo=0.9)

    model_low = Cigre207(
        example_span_1_conductor,
        weather_low_albedo,
        when,
        include_diffuse_radiation=False,
    )
    model_high = Cigre207(
        example_span_1_conductor,
        weather_high_albedo,
        when,
        include_diffuse_radiation=False,
    )

    assert (
        model_low.compute_global_radiation_intensity()
        == model_high.compute_global_radiation_intensity()
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_compute_global_radiation_intensity_scales_with_direct_factor_without_diffuse(
    example_span_1_conductor: Span, example_weather_a: Weather, when: Date
):
    model_half = Cigre207(
        example_span_1_conductor,
        example_weather_a,
        when,
        include_diffuse_radiation=False,
        direct_radiation_factor=0.5,
    )
    model_full = Cigre207(
        example_span_1_conductor,
        example_weather_a,
        when,
        include_diffuse_radiation=False,
        direct_radiation_factor=1.0,
    )

    assert 2.0 * model_half.compute_global_radiation_intensity() == approx(
        model_full.compute_global_radiation_intensity()
    )


@hypothesis.settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@hypothesis.given(when=numpy_datetimes())
def test_compute_global_radiation_intensity_is_zero_with_zero_direct_factor_and_no_diffuse(
    example_span_1_conductor: Span, example_weather_a: Weather, when: Date
):
    model = Cigre207(
        example_span_1_conductor,
        example_weather_a,
        when,
        include_diffuse_radiation=False,
        direct_radiation_factor=0.0,
    )

    assert model.compute_global_radiation_intensity() == 0.0


def test_global_radiation_intensity_example(
    example_span_1_conductor: Span, example_weather_a: Weather
):
    when = np.datetime64("2022-03-12T12:20")
    model = Cigre207(example_span_1_conductor, example_weather_a, when)
    assert model.compute_global_radiation_intensity() == 1228.3849602842813
