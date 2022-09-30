from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import pygeodesy

from .units import (
    Celsius,
    Degrees,
    Meter,
    MeterPerSecond,
    OhmPerMeter,
    PerCelsius,
    PerSquareCelsius,
    Radian,
    SquareMeter,
    SquareMeterPerAmpere,
    Unitless,
    WattPerMeterPerKelvin,
)

__all__ = ["Conductor", "Weather", "Tower", "Span"]


@dataclass
class Conductor:
    core_diameter: Meter
    conductor_diameter: Meter
    outer_layer_strand_diameter: Meter

    emissivity: Unitless
    solar_absorptivity: Unitless

    resistance_at_20c: OhmPerMeter
    linear_resistance_coefficient_20c: PerCelsius
    quadratic_resistance_coefficient_20c: PerSquareCelsius

    aluminium_surface_area: SquareMeter
    constant_magnetic_effect: Unitless
    current_density_proportional_magnetic_effect: SquareMeterPerAmpere
    max_magnetic_core_relative_resistance_increase: Unitless

    thermal_conductivity: Optional[WattPerMeterPerKelvin] = None


@dataclass
class Tower:
    longitude: Degrees
    latitude: Degrees
    altitude: Meter


@dataclass
class Span:
    conductor: Conductor
    start_tower: Tower
    end_tower: Tower
    ground_albedo: Unitless
    num_conductors: Unitless

    @cached_property
    def latitude(self) -> Degrees:
        return 0.5 * (self.start_tower.latitude + self.end_tower.latitude)

    @cached_property
    def longitude(self) -> Degrees:
        return 0.5 * (self.start_tower.longitude + self.end_tower.longitude)

    @cached_property
    def inclination(self) -> Radian:
        delta_y = np.abs(self.end_tower.altitude - self.start_tower.altitude)
        return np.arctan2(delta_y, self.span_length)

    @cached_property
    def conductor_azimuth(self) -> Radian:
        return np.radians(  # type: ignore
            pygeodesy.formy.bearing(
                lat1=self.start_tower.latitude,
                lon1=self.start_tower.longitude,
                lat2=self.end_tower.latitude,
                lon2=self.end_tower.longitude,
            )
        )

    @cached_property
    def span_length(self) -> Meter:
        return pygeodesy.formy.haversine(  # type: ignore
            lat1=self.start_tower.latitude,
            lon1=self.start_tower.longitude,
            lat2=self.end_tower.latitude,
            lon2=self.end_tower.longitude,
        )

    @cached_property
    def conductor_altitude(self) -> Meter:
        return 0.5 * (self.start_tower.altitude + self.end_tower.altitude)


@dataclass
class Weather:
    air_temperature: Celsius
    wind_direction: Radian  # TODO: Consider inputing wind east and wind north
    wind_speed: MeterPerSecond
    clearness_ratio: Unitless = 1
