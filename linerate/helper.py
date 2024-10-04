from typing import Optional

import numpy as np
import pvlib

from linerate import Conductor, Span, Tower, Weather
from linerate.conductor_params_finder import ConductorFinder, ConductorFinderProtocol
from linerate.models.cigre207 import Cigre207
from linerate.models.cigre601 import Cigre601
from linerate.models.ieee738 import IEEE738

model_mapping = {
    "Cigre601": Cigre601,
    "IEEE738": IEEE738,
    "Cigre207": Cigre207
}

def compute_line_rating(dataframe,
                       model_name = "Cigre601",
                       wind_speed_min = 0.61,
                       angle_of_attack_low_speed_threshold = 2.0,
                       angle_of_attack_target_angle = 45.0):
    """
    Easy access function with default conductor parameters.
    See LineRatingComputation.compute_line_rating_from_dataframe for parameter details.
    """
    lrc = LineRatingComputation()
    return lrc.compute_line_rating_from_dataframe(
        dataframe,
        model_name,
        wind_speed_min,
        angle_of_attack_low_speed_threshold,
        angle_of_attack_target_angle
    )

def calculate_solar_irradiance(latitude, longitude, timestamps):
    """
    Calculate the solar irradiance at a given location and time with clear sky assumption.
    When using pandas dataframes, be careful about indexing by the timestamps. See test_helper.py for an example.

    Parameters
    ----------
    latitude
    longitude
    timestamps

    Returns
    -------
    pd.Series with irradiance values

    """
    # Get solar position
    solpos = pvlib.solarposition.get_solarposition(timestamps, latitude, longitude)

    # Calculate clear-sky irradiance
    irradiance = pvlib.clearsky.haurwitz(solpos['apparent_zenith'])

    return irradiance


class LineRatingComputation:

    def __init__(self, conductor_finder: Optional[ConductorFinderProtocol] = ConductorFinder()):
        self.conductor_finder = conductor_finder

    def compute_line_rating_from_dataframe(self,
                                           dataframe,
                                           model_name = "Cigre601",
                                           wind_speed_min = 0.61,
                                           angle_of_attack_low_speed_threshold = 2.0,
                                           angle_of_attack_target_angle = 45.0):
        """
        Calculate the Line Rating for all rows in dataframe.

        :param dataframe: pd.DataFrame
            A pandas DataFrame containing spans and related weather data.
            The DataFrame must have the following columns:
            **General columns**
            - `line_id` (int)
            - `span_number` (int)
            - `timestamp` (datetime)
            **Weather-related columns**
            - `temperature` (float)
            - `wind_speed` (float)
            - `wind_direction` (float)
            - `humidity` (float)
            - `solar_radiation_clouds` (float)
            **Span-related columns**
            - `start_lon` (float)
            - `start_lat` (float)
            - `end_lon` (float)
            - `end_lat` (float)
            - `mid_lon` (float)
            - `mid_lat` (float)
            - `bearing` (float)
            - `wires` (int)
            - `max_allowed_temp` (float)
            - `conductor` (str, conductor name)
            **Optional column**
            - `elevation` (float) (default=0)
            - `ground_albedo` (float) (default=0.15)
            - `emissivity` (float) (default=0.9)
            - `solar_absorptivity` (float) (default=0.9)
            - `constant_magnetic_effect` (float) (default=1)
            - `current_density_proportional_magnetic_effect` (float) (default=0)
            - `max_magnetic_core_relative_resistance_increase` (float) (default=1)

        :param model_name:
            One of: `'Cigre601'`, `'IEEE738'`, `'Cigre207'` (default is `'Cigre601'`).
        :param wind_speed_min:
            Minimum wind speed value to clip the wind speed to, in meters per second (default is 0.61).
        :param angle_of_attack_low_speed_threshold:
            Wind speed threshold (float) for attack angle calculation. Speeds lower than this threshold will be set to `angle_of_attack_target_angle` (default is 2.0).
        :param angle_of_attack_target_angle:
            Target angle of attack (float, in degrees) to use when wind speed is below `angle_of_attack_low_speed_threshold` (default is 45.0).

        :return:
            A pandas Series containing Ampere values for each row in the input dataframe.
        :rtype: pd.Series
        """
        required_columns_df = [
            'line_id', 'span_number', 'timestamp',
            # Weather related columns
            'temperature', 'wind_speed', 'wind_direction', 'humidity', 'solar_radiation_clouds',
            # Span related columns
            'start_lon', 'start_lat', 'end_lon', 'end_lat',
            'mid_lon', 'mid_lat', 'bearing', 'wires', 'max_allowed_temp', 'conductor'
        ]

        missing_columns = [column for column in required_columns_df if column not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Input DataFrame is missing mandatory columns: {missing_columns}")
        model = model_mapping[model_name]

        if 'elevation' not in dataframe.columns:
            dataframe['elevation'] = 0
        if 'ground_albedo' not in dataframe.columns:
            dataframe['ground_albedo'] = 0.15
        if 'emissivity' not in dataframe.columns:
            dataframe['emissivity'] = 0.9
        if 'solar_absorptivity' not in dataframe.columns:
            dataframe['solar_absorptivity'] = 0.9
        if 'constant_magnetic_effect' not in dataframe.columns:
            dataframe['constant_magnetic_effect'] = 1
        if 'current_density_proportional_magnetic_effect' not in dataframe.columns:
            dataframe['current_density_proportional_magnetic_effect'] = 0
        if 'max_magnetic_core_relative_resistance_increase' not in dataframe.columns:
            dataframe['max_magnetic_core_relative_resistance_increase'] = 1

        conductor_params = self.conductor_finder.find_conductor_parameters_by_names(dataframe['conductor'])

        conductor = Conductor(
            core_diameter=conductor_params['core_diameter_mm'] * 1e-3,
            conductor_diameter=conductor_params['conductor_diameter_mm'] * 1e-3,
            outer_layer_strand_diameter=conductor_params['wire_diameter_al_mm'] * 1e-3,
            emissivity=dataframe['emissivity'], # not conductor specific, set by Transmission System Operator (TSO)
            solar_absorptivity=dataframe['solar_absorptivity'], # not conductor specific, set by Transmission System Operator (TSO)
            temperature1=conductor_params['low_temperature_deg_c'],
            temperature2=conductor_params['high_temperature_deg_c'],
            resistance_at_temperature1=conductor_params['dc_resistance_low_temperature_ohm_per_km'] * 1e-3,
            resistance_at_temperature2=conductor_params['dc_resistance_high_temperature_ohm_per_km'] * 1e-3,
            aluminium_cross_section_area=conductor_params['area_al_sq_mm'] * 1e-6, # Convert mm2 to m2
            constant_magnetic_effect=dataframe['constant_magnetic_effect'], # not conductor specific, set by TSO
            current_density_proportional_magnetic_effect=dataframe['current_density_proportional_magnetic_effect'], # not conductor specific, set by TSO
            max_magnetic_core_relative_resistance_increase=dataframe['max_magnetic_core_relative_resistance_increase'], # not conductor specific, set by TSO
        )

        span = Span(
            conductor=conductor,
            start_tower=Tower(latitude=dataframe['start_lat'], longitude=dataframe['start_lon'], altitude=dataframe['elevation']),
            end_tower=Tower(latitude=dataframe['end_lat'], longitude=dataframe['end_lon'], altitude=dataframe['elevation']),
            latitude=dataframe['mid_lat'],
            longitude=dataframe['mid_lon'],
            bearing=dataframe['bearing'],
            ground_albedo=dataframe['ground_albedo'],
            num_conductors=dataframe['wires'],
        )

        # Clip wind speed
        dataframe['wind_speed'] = np.where(dataframe['wind_speed'] < wind_speed_min, wind_speed_min, dataframe['wind_speed'])

        weather = Weather(
            air_temperature=dataframe['temperature'],
            wind_direction=dataframe['wind_direction'],
            wind_speed=dataframe['wind_speed'],
            clearness_ratio=0,
            solar_irradiance=dataframe['solar_radiation_clouds'],
        )

        model_instance = model(
            span=span,
            weather=weather,
            time=dataframe['timestamp'],
            angle_of_attack_low_speed_threshold=angle_of_attack_low_speed_threshold,
            angle_of_attack_target_angle=angle_of_attack_target_angle
        )

        # Calculate the steady-state ampacity
        return model_instance.compute_steady_state_ampacity(dataframe['max_allowed_temp'])