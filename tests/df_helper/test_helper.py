import os
import numpy as np
import pandas as pd
import pytest
import linerate
from linerate.helper import LineRatingComputation, compute_line_rating, calculate_solar_irradiance

def test_example_helper_usage():
    df = pd.DataFrame({
        'line_id': [1],
        'span_number': ['45.6789_50.1234'],
        'timestamp': [np.datetime64('2024-01-01T00:00:00')],
        'temperature': [20.0],
        'wind_speed': [5.0],
        'wind_direction': [90],
        'humidity': [50.0],
        'solar_radiation_clouds': [800.0],
        'start_lon': [24.9384],
        'start_lat': [60.1699],
        'end_lon': [24.9484],
        'end_lat': [60.1799],
        'mid_lon': [24.9434],
        'mid_lat': [60.1749],
        'bearing': [45.0],
        'wires': [1],
        'max_allowed_temp': [60.0],
        'conductor': ['565-AL1/72-ST1A']
    })
    result = compute_line_rating(df)
    assert isinstance(result, pd.Series)


def _pickle_path(name):
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, 'fixtures', name)

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
        'span_number': ['45.6789_50.1234'],
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


def test_simplest_helper(example_dataframe_b):
    # Use this, when using default conductor parameters.
    result = compute_line_rating(example_dataframe_b)
    assert isinstance(result, pd.Series)

def test_dataframe_helper_works(example_dataframe_b):
    # LineRatingComputation should when it can be re-used
    helper = LineRatingComputation()
    result = helper.compute_line_rating_from_dataframe(example_dataframe_b)
    assert isinstance(result, pd.Series)

def test_helper_with_missing_column_throws_error(example_dataframe_b):
    df = example_dataframe_b.copy()
    d = df.drop(columns=['temperature'])
    helper = LineRatingComputation()
    with pytest.raises(ValueError):
        helper.compute_line_rating_from_dataframe(d)

@pytest.fixture
def fingrid_conductor_finder():
    class InlineConductorFinder:
        @staticmethod
        def find_conductor_parameters_by_names(name_series: pd.Series) -> pd.DataFrame:
            fixed_params = {
                'code': 'fixed_code',
                'old_code': 'fixed_old_code',
                'area_al_sq_mm': 565.0,
                'area_steel_sq_mm': 71.6,
                'area_total_sq_mm': 636.6,
                'no_of_wires_al': 54,
                'no_of_wires_steel': 19,
                'wire_diameter_al_mm': 3.65,
                'wire_diameter_steel_mm': 3.65,
                'core_diameter_mm': 11.0,
                'conductor_diameter_mm': 32.9,
                'mass_per_unit_length_kg_per_km': 14.9,
                'rated_strength_kn': 174.14,
                'dc_resistance_low_temperature_ohm_per_km': 0.0500,
                'final_modulus_of_elasticity_n_per_sq_mm': 76000,
                'coefficient_of_linear_expansion_per_k': 0.000023,
                'current_carrying_capacity_a': 0,
                'country': 'FI',
                'aluminium_type': 'AL1',
                'low_temperature_deg_c': 20,
                'high_temperature_deg_c': 80,
                'dc_resistance_high_temperature_ohm_per_km': 0.0621
            }
            return pd.DataFrame([fixed_params] * len(name_series))

    return InlineConductorFinder()

def test_dataframe_helper_on_v101_data_dlr(fingrid_conductor_finder):
    input_df = pd.read_pickle(_pickle_path('dlr_comp_input_1.0.1.pkl'))
    previous_result_dlr = pd.read_pickle(_pickle_path('dlr_comp_output_1.0.1.pkl'))
    helper = LineRatingComputation(fingrid_conductor_finder)
    result = helper.compute_line_rating_from_dataframe(input_df, angle_of_attack_low_speed_threshold=3)
    diff = previous_result_dlr.compare(result)
    assert previous_result_dlr.equals(result), f"DataFrames are not equal:\n{diff}"

def test_dataframe_helper_on_v101_data_slr(fingrid_conductor_finder):
    input_df = pd.read_pickle(_pickle_path('dlr_comp_input_1.0.1.pkl'))

    input_df['solar_radiation_clouds'] = 1033
    input_df['wind_speed'] = 0.61
    input_df['temperature'] = 25
    # wind direction in 90-degree angle
    input_df['wind_direction'] = (input_df['bearing'] + 90) % 360

    previous_result = pd.read_pickle(_pickle_path('slr_comp_output_1.0.1.pkl'))
    helper = LineRatingComputation(fingrid_conductor_finder)
    result = helper.compute_line_rating_from_dataframe(input_df, angle_of_attack_low_speed_threshold=3)
    diff = previous_result.compare(result)
    assert previous_result.equals(result), f"DataFrames are not equal:\n{diff}"


def test_calculate_solar_irradiance():
    datetime_idx = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 23:00:00', freq='h', tz='UTC')
    df = pd.DataFrame({'timestamp': datetime_idx})
    df['latitude'] = 60.1699
    df['longitude'] = 24.9384

    # Using this function, the dataframe has to be indexed according to timestamp
    df.set_index('timestamp', inplace=True)
    result = calculate_solar_irradiance(df['latitude'], df['longitude'], df.index)
    df['irradiance'] = result
    df.reset_index(inplace=True)


def test_calculate_solar_irradiance_empty_timestamps():
    latitude = pd.Series([])
    longitude = pd.Series([])
    timestamps = pd.DatetimeIndex([])

    result = calculate_solar_irradiance(latitude, longitude, timestamps)

    assert result.empty == True