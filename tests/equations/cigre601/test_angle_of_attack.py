import datetime

import numpy as np
from linerate.models.cigre601 import Cigre601
from linerate.types import Span, Weather


def __create_span(bearing: float) -> Span:
    return Span(bearing=bearing, conductor=None, start_tower=None, end_tower=None, latitude=0, longitude=0, ground_albedo=0, num_conductors=1)


def test_compute_angle_of_attack_below_clip_within_range():
    span = __create_span(30)
    weather = Weather(
        air_temperature=25,
        wind_direction=20,
        wind_speed=2,
        solar_irradiance=800
    )
    model = Cigre601(span=span,
                     weather=weather,
                     time=np.datetime64(datetime.datetime.now()),
                     angle_of_attack_low_speed_threshold=3,
                     angle_of_attack_target_angle=45)
    angle_of_attack = model.compute_angle_of_attack()
    assert np.isclose(angle_of_attack, np.radians(45))

def test_compute_angle_of_attack_below_clip_outside_range():
    span = __create_span(30)
    weather = Weather(
        air_temperature=25,
        wind_direction=200,
        wind_speed=2,
        solar_irradiance=800
    )
    model = Cigre601(span=span,
                     weather=weather,
                     time=np.datetime64(datetime.datetime.now()),
                     angle_of_attack_low_speed_threshold=3,
                     angle_of_attack_target_angle=45)
    angle_of_attack = model.compute_angle_of_attack()
    assert np.isclose(angle_of_attack, np.radians(135))

def test_compute_angle_of_attack_above_clip():
    span = __create_span(30)
    weather = Weather(
        air_temperature=25,
        wind_direction=200,
        wind_speed=5,
        solar_irradiance=800
    )
    model = Cigre601(span=span,
                     weather=weather,
                     time=np.datetime64(datetime.datetime.now()),
                     angle_of_attack_low_speed_threshold=3,
                     angle_of_attack_target_angle=45)
    angle_of_attack = model.compute_angle_of_attack()
    expected_angle = np.abs(200 - 30)
    if expected_angle > 180:
        expected_angle = 360 - expected_angle
    assert np.isclose(angle_of_attack, np.radians(expected_angle))

def test_compute_angle_of_attack_boundary_conditions():
    span = __create_span(0)
    weather = Weather(
        air_temperature=25,
        wind_direction=180,
        wind_speed=2,
        solar_irradiance=800
    )
    model = Cigre601(span=span,
                     weather=weather,
                     time=np.datetime64(datetime.datetime.now()),
                     angle_of_attack_low_speed_threshold=3,
                     angle_of_attack_target_angle=45)
    angle_of_attack = model.compute_angle_of_attack()
    assert np.isclose(angle_of_attack, np.radians(135))