import pytest
import linerate.helper as helper
from linerate.conductor_params_finder import ConductorFinder
import pandas as pd
import numpy as np

# Default values for the test dataframe
__defaults = {
    'temperature': 35.0,
    'wind_speed': 0.6,
    'wind_direction': 90,
    'humidity': 100.0,
    'solar_radiation': 1033.0,
    'max_allowed_temp': 80.0,
    'conductor': '565-AL1/72-ST1A',
    'wires': [1]
}

def create_test_dataframe(**kwargs):
    params = {**__defaults, **kwargs}
    return pd.DataFrame({
        'line_id': [1],
        'span_number': ['45.6789_50.1234'],
        'timestamp': [np.datetime64('2024-01-01T00:00:00')],
        'temperature': [params['temperature']],
        'wind_speed': [params['wind_speed']],
        'wind_direction': [params['wind_direction']],
        'humidity': [params['humidity']],
        'solar_radiation_clouds': [params['solar_radiation']],
        'start_lon': [24.9384],
        'start_lat': [60.1699],
        'end_lon': [24.9484],
        'end_lat': [60.1799],
        'mid_lon': [24.9434],
        'mid_lat': [60.1749],
        'bearing': [0],
        'wires': params['wires'],
        'max_allowed_temp': [params['max_allowed_temp']],
        'conductor': [params['conductor']]
    })

def test_nominal_case():
    df = create_test_dataframe()
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

@pytest.mark.parametrize("conductor", [
    '565-AL1/72-ST1A', '402-AL1/52-ST1A', '242-AL1/39-ST1A', '152-A1/S1A-26/7'
])
def test_estimated_rating(conductor):
    finder = ConductorFinder()
    expected_value = finder.find_conductor_parameters_by_names(pd.Series(conductor)).iloc[0].current_carrying_capacity_a
    df = create_test_dataframe(conductor=conductor)
    result = helper.compute_line_rating(df)
    tolerance = 0.05 * expected_value
    lower_bound = expected_value - tolerance
    upper_bound = expected_value + tolerance
    computed_rating = result.iloc[0]
    assert lower_bound <= computed_rating <= upper_bound, f"Line rating of {conductor} = {computed_rating} is not within the expected range of {lower_bound} to {upper_bound}"

@pytest.mark.parametrize("temperature", [-40.0, 0, 35.0])
def test_temperature_range(temperature):
    df = create_test_dataframe(temperature=temperature)
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

@pytest.mark.parametrize("wires", [1, 2, 3, 4])
def test_wires_range(wires):
    df = create_test_dataframe(wires=wires)
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

@pytest.mark.parametrize("wind_speed", [0.0, 5.0, 20.0])
def test_wind_speed_range(wind_speed):
    df = create_test_dataframe(wind_speed=wind_speed)
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

@pytest.mark.parametrize("wind_direction", [0.0, 90.0, 180.0, 360.0])
def test_wind_direction_range(wind_direction):
    df = create_test_dataframe(wind_direction=wind_direction)
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

@pytest.mark.parametrize("solar_radiation", [0.0, 500.0, 1200.0])
def test_solar_radiation_range(solar_radiation):
    df = create_test_dataframe(solar_radiation=solar_radiation)
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

@pytest.mark.parametrize("max_allowed_temp", [60, 80.0, 100])
def test_max_allowed_temp_range(max_allowed_temp):
    df = create_test_dataframe(max_allowed_temp=max_allowed_temp)
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

@pytest.mark.parametrize("conductor", ['565-AL1/72-ST1A', '402-AL1/52-ST1A', '242-AL1/39-ST1A', '152-A1/S1A-26/7'])
def test_different_conductor_types(conductor):
    df = create_test_dataframe(conductor=conductor)
    result = helper.compute_line_rating(df)
    assert result is not None
    assert not result.empty

def test_increasing_wires_increases_rating():
    df1 = create_test_dataframe(wires=1)
    df2 = create_test_dataframe(wires=2)
    result1 = helper.compute_line_rating(df1).iloc[0]
    result2 = helper.compute_line_rating(df2).iloc[0]
    assert result2 > result1, "Line rating should increase as number of wires increase"

def test_increasing_temperature_decreases_rating():
    df1 = create_test_dataframe(temperature=20.0)
    df2 = create_test_dataframe(temperature=40.0)
    result1 = helper.compute_line_rating(df1).iloc[0]
    result2 = helper.compute_line_rating(df2).iloc[0]
    assert result2 < result1, "Line rating should decrease as temperature increases"

def test_increasing_wind_speed_increases_rating():
    df1 = create_test_dataframe(wind_speed=0.5)
    df2 = create_test_dataframe(wind_speed=5.0)
    result1 = helper.compute_line_rating(df1).iloc[0]
    result2 = helper.compute_line_rating(df2).iloc[0]
    assert result2 > result1, "Line rating should increase as wind speed increases"

def test_decreasing_solar_radiation_increases_rating():
    df1 = create_test_dataframe(solar_radiation=1033.0)
    df2 = create_test_dataframe(solar_radiation=500.0)
    result1 = helper.compute_line_rating(df1).iloc[0]
    result2 = helper.compute_line_rating(df2).iloc[0]
    assert result2 > result1, "Line rating should increase as solar radiation decreases"

def test_wind_direction_effect_on_rating():
    df1 = create_test_dataframe(wind_direction=0.0)
    df2 = create_test_dataframe(wind_direction=90.0)
    result1 = helper.compute_line_rating(df1).iloc[0]
    result2 = helper.compute_line_rating(df2).iloc[0]
    assert result2 > result1, "Line rating should increase with crosswind compared to headwind"