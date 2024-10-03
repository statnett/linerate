import numpy as np
import pandas as pd
import pytest

import linerate
from linerate.helper import LineRatingComputation

@pytest.fixture
def drake_conductor_b():
    return linerate.Conductor(
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
def example_weather_b():
    return linerate.Weather(
        air_temperature=20,
        wind_direction=np.radians(80),
        wind_speed=1.66,
        clearness_ratio=0.5,
        solar_irradiance=0.5,
    )


@pytest.fixture()
def example_span_b(drake_conductor_b):
    start_tower = linerate.Tower(latitude=50 - 0.0045, longitude=0, altitude=500 - 88)
    end_tower = linerate.Tower(latitude=50 + 0.0045, longitude=0, altitude=500 + 88)
    return linerate.Span(
        conductor=drake_conductor_b,
        start_tower=start_tower,
        end_tower=end_tower,
        latitude=50,
        longitude=0,
        bearing=calculate_bearing(start_tower.latitude, start_tower.longitude, end_tower.latitude, end_tower.longitude),
        ground_albedo=0.15,
        num_conductors=1,
    )

@pytest.fixture()
def example_dataframe_b(example_span_b, example_weather_b):
    #  # General columns
    #  - 'line_id': Int
    #  - 'span_number': int
    #  - 'timestamp': datetime
    #  # Weather related columns
    #  - 'temperature': float
    #  - 'wind_speed': float
    #  - 'wind_direction': float - radians
    #  - 'humidity': float
    #  - 'solar_radiation_clouds': float
    #  # Span related columns
    #  - 'start_lon': Float
    #  - 'start_lat': Float
    #  - 'end_lon': Float
    #  - 'end_lat': Float
    #  - 'mid_lon': Float
    #  - 'mid_lat': Float
    #  - 'bearing': Float
    #  - 'wires': Int
    #  - 'max_allowed_temp': Float
    #  - 'conductor': str (conductor name)
    input_params = {
        "timestamp": np.datetime64("2016-10-03 14:00"),
        "max_allowed_temp": 60,
        "conductor": "565-AL1/72-ST1A"
    }


    bearings = calculate_bearing(example_span_b.start_tower.latitude,
                                                            example_span_b.start_tower.longitude,
                                                            example_span_b.end_tower.latitude,
                                                            example_span_b.end_tower.longitude)
    # Calculate midpoint
    mid_lat = (example_span_b.start_tower.latitude + example_span_b.end_tower.latitude) / 2
    mid_lon = (example_span_b.start_tower.longitude + example_span_b.end_tower.longitude) / 2


    df = pd.DataFrame({
        'line_id': [1],
        'span_number': [1],
        'timestamp': [input_params['timestamp']],
        'temperature': [example_weather_b.air_temperature],
        'wind_speed': [example_weather_b.wind_speed],
        'wind_direction': [example_weather_b.wind_direction],
        'humidity': [0],
        'solar_radiation_clouds': [example_weather_b.solar_irradiance],
        'start_lon': [example_span_b.start_tower.longitude],
        'start_lat': [example_span_b.start_tower.latitude],
        'end_lon': [example_span_b.end_tower.longitude],
        'end_lat': [example_span_b.end_tower.latitude],
        'mid_lon': [mid_lon],
        'mid_lat': [mid_lat],
        'bearing': bearings,
        'wires': [example_span_b.num_conductors],
        'max_allowed_temp': [input_params["max_allowed_temp"]],
        'conductor': [input_params["conductor"]],
    })
    return df

def calculate_bearing(lat1, lon1, lat2, lon2):
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    rlon1 = np.radians(lon1)
    rlon2 = np.radians(lon2)
    dlon = rlon2 - rlon1

    x = np.cos(rlat2) * np.sin(dlon)
    y = np.cos(rlat1) * np.sin(rlat2) - np.sin(rlat1) * np.cos(rlat2) * np.cos(dlon)
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def test_dataframe_helper_works(example_dataframe_b):
    helper = LineRatingComputation()
    result = helper.compute_line_rating_from_dataframe(example_dataframe_b)
    assert isinstance(result, pd.Series)

def test_helper_with_missing_column_throws_error(example_dataframe_b):
    df = example_dataframe_b.copy()
    d = df.drop(columns=['temperature'])
    helper = LineRatingComputation()
    with pytest.raises(ValueError):
        helper.compute_line_rating_from_dataframe(d)


def test_dataframe_helper_on_v101_data_dlr():
    input_df = pd.read_pickle('fixtures/dlr_comp_input_1.0.1.pkl')
    previous_result_dlr = pd.read_pickle('fixtures/dlr_comp_output_1.0.1.pkl')
    helper = LineRatingComputation()
    result = helper.compute_line_rating_from_dataframe(input_df, angle_of_attack_low_speed_threshold=3)
    diff = previous_result_dlr.compare(result)
    assert previous_result_dlr.equals(result), f"DataFrames are not equal:\n{diff}"

def test_dataframe_helper_on_v101_data_slr():
    input_df = pd.read_pickle('fixtures/dlr_comp_input_1.0.1.pkl')

    input_df['solar_radiation_clouds'] = 1033
    input_df['wind_speed'] = 0.61
    input_df['temperature'] = 25
    # wind direction in 90-degree angle
    input_df['wind_direction'] = (input_df['bearing'] + 90) % 360

    previous_result = pd.read_pickle('fixtures/slr_comp_output_1.0.1.pkl')
    helper = LineRatingComputation()
    result = helper.compute_line_rating_from_dataframe(input_df, angle_of_attack_low_speed_threshold=3)
    diff = previous_result.compare(result)
    assert previous_result.equals(result), f"DataFrames are not equal:\n{diff}"
