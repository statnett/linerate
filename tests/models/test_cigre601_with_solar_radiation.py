import numpy as np

from linerate.equations import cigre601, solar_angles
from linerate.models.cigre601 import Cigre601, Cigre601WithSolarRadiation
from linerate.types import WeatherWithSolarRadiation


def get_weather_with_solar_radiation(
    model: Cigre601,
) -> WeatherWithSolarRadiation:
    sin_H_s = solar_angles.compute_sin_solar_altitude_for_span(model.span, model.time)
    y = model.span.conductor_altitude
    N_s = model.weather.clearness_ratio

    I_B = cigre601.solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y)
    I_d = cigre601.solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)

    return WeatherWithSolarRadiation(
        air_temperature=model.weather.air_temperature,
        wind_direction=model.weather.wind_direction,
        wind_speed=model.weather.wind_speed,
        ground_albedo=model.weather.ground_albedo,
        direct_radiation_intensity=I_B,
        diffuse_radiation_intensity=I_d,
    )


def test_solar_heating_with_solar_radiation_equals_with_cigre601_radiation(
    example_model_1_conductors: Cigre601,
):
    weather_with_radiation = get_weather_with_solar_radiation(example_model_1_conductors)
    model_with_radiation = Cigre601WithSolarRadiation(
        example_model_1_conductors.span, weather_with_radiation, example_model_1_conductors.time
    )

    np.testing.assert_allclose(
        model_with_radiation.compute_solar_heating(np.nan, np.nan),
        example_model_1_conductors.compute_solar_heating(np.nan, np.nan),
    )


def test_ampacity_with_solar_radiation_equals_with_cigre601_radiation(
    example_model_1_conductors: Cigre601,
):
    weather_with_radiation = get_weather_with_solar_radiation(example_model_1_conductors)
    model_with_radiation = Cigre601WithSolarRadiation(
        example_model_1_conductors.span, weather_with_radiation, example_model_1_conductors.time
    )

    np.testing.assert_allclose(
        model_with_radiation.compute_steady_state_ampacity(90.0),
        example_model_1_conductors.compute_steady_state_ampacity(90.0),
    )
