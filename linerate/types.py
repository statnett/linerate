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
    Radian,
    SquareMeter,
    SquareMeterPerAmpere,
    Unitless,
    WattPerMeterPerKelvin,
)

__all__ = ["Conductor", "Weather", "Tower", "Span"]


@dataclass(frozen=True)
class Conductor:
    """Container for conductor parameters."""

    #: :math:`D_1~\left[\text{m}\right]`. Diameter of the steel core of the conductor.
    core_diameter: Meter

    #: :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    conductor_diameter: Meter

    #: :math:`d~\left[\text{m}\right]`. The diameter of the strands in the outer layer of
    #: the conductor.
    outer_layer_strand_diameter: Meter

    #: :math:`\epsilon_s`. The emmisivity of the conductor.
    emissivity: Unitless

    #: :math:`\alpha_s`. Material constant. According to :cite:p:`cigre601`, it starts at
    #: approximately 0.2 for new cables and reaches a constant value of approximately 0.9
    #: after about one year.
    solar_absorptivity: Unitless

    #: :math:`T_1~\left[^\circ C\right]`. The first temperature with known resistance
    temperature1: Celsius
    #: :math:`T_2~\left[^\circ C\right]`. The second temperature with known resistance
    temperature2: Celsius
    #: :math:`R_1~\left[\Omega \text{m}^{-1}\right]`. (AC-)resistance at temperature :math:`T_1`
    resistance_at_temperature1: OhmPerMeter
    #: :math:`R_2~\left[\Omega \text{m}^{-1}\right]`. (AC-)resistance at temperature :math:`T_2`
    resistance_at_temperature2: OhmPerMeter

    #: :math:`A_{\text{Al}}~\left[\text{m}^2\right]`. The cross sectional area of the aluminium
    #: strands in the conductor. Used for correcting for magnetic core effects in ACSR conductors.
    aluminium_cross_section_area: SquareMeter
    #: :math:`b`. The constant magnetic effect, most likely equal to 1. If ``None``, then no
    #: correction is used (used for non-ACSR cables).
    constant_magnetic_effect: Unitless
    #: :math:`m`. The current density proportional magnetic effect. If ``None``, then it is assumed
    #: equal to 0.
    current_density_proportional_magnetic_effect: SquareMeterPerAmpere
    #: :math:`c_\text{max}`. Saturation point of the relative increase in conductor resistance due
    # to magnetic core effects.
    max_magnetic_core_relative_resistance_increase: Unitless

    #: :math:`\lambda \left[\text{W}~\text{m}^{-1}~\text{K}^{-1}\right]`. The effective
    #: conductor thermal conductivity. It is usually between :math:`0.5` and
    #: :math:`7~W~m^{-1}~K^{-1}`. Recommended values are
    #: :math:`0.7~\text{W}~\text{m}^{-1}~\text{K}^{-1}` for conductors with no tension on the
    #: aluminium strands and :math:`1.5~\text{W}~\text{m}^{-1}~\text{K}^{-1}` for conductors
    #: with aluminium strands under a tension of at least 40 N :cite:p:`cigre601`.
    thermal_conductivity: Optional[WattPerMeterPerKelvin] = None


@dataclass(frozen=True)
class Tower:
    """Container for a tower (span end-point)."""

    #: :math:`\phi~\left[^\circ\right]`. The tower's longitude (east of the prime meridian).
    longitude: Degrees
    #: The tower's latitude (north-facing).
    latitude: Degrees
    #: :math:`y~\left[m\right]`. The tower's altitude.
    altitude: Meter


@dataclass(frozen=True)
class Span:
    """Container for a span.

    Note
    ----
    For more information about the albedo, see
    :py:func:`linerate.equations.solar_heating.compute_global_radiation_intensity` for a table of
    albedo values for different ground types.
    """

    #: Container for the conductor metadata
    conductor: Conductor
    #: Container for the metadata of the first tower of the span
    start_tower: Tower
    #: Container for the metadata of the second tower of the span
    end_tower: Tower

    #: :math:`F`.  The ground albedo.
    ground_albedo: Unitless

    #: Number of conductors in the span. 1 for simplex, 2 for duplex and 3 for triplex.
    num_conductors: Unitless

    @cached_property
    def latitude(self) -> Degrees:
        r""":math:`\phi~\left[^\circ\right]`. The latitude of the span midpoint."""
        return 0.5 * (self.start_tower.latitude + self.end_tower.latitude)

    @cached_property
    def longitude(self) -> Degrees:
        r""":math:`\left[^\circ\right]`. The longitude of the span midpoint."""
        return 0.5 * (self.start_tower.longitude + self.end_tower.longitude)

    @cached_property
    def inclination(self) -> Radian:
        r""":math:`\beta~\left[\text{radian}\right]`. The inclination.

        The inclination is computed from the difference in span altitude and the span length.
        """
        delta_y = np.abs(self.end_tower.altitude - self.start_tower.altitude)
        return np.arctan2(delta_y, self.span_length)

    @cached_property
    def conductor_azimuth(self) -> Radian:
        r""":math:`\gamma_c~\left[\text{radian}\right]`. Angle (east of north) the span is facing"""
        bearing = np.vectorize(pygeodesy.formy.bearing)
        return np.radians(  # type: ignore
            bearing(
                lat1=self.start_tower.latitude,
                lon1=self.start_tower.longitude,
                lat2=self.end_tower.latitude,
                lon2=self.end_tower.longitude,
            )
        )

    @cached_property
    def span_length(self) -> Meter:
        r""":math:`\left[\text{km}\right]`. The span length.

        The span length is computed with the haversine formula (assuming spherical earth).
        """
        haversine = np.vectorize(pygeodesy.formy.haversine)
        return haversine(  # type: ignore
            lat1=self.start_tower.latitude,
            lon1=self.start_tower.longitude,
            lat2=self.end_tower.latitude,
            lon2=self.end_tower.longitude,
        )

    @cached_property
    def conductor_altitude(self) -> Meter:
        r""":math:`y~\left[\text{m}\right]`. The span altitude.

        The altitude is computes as the average of the tower altitudes.
        """
        return 0.5 * (self.start_tower.altitude + self.end_tower.altitude)


@dataclass()
class Weather:
    #: :math:`T_a~\left[^\circ C\right]`. The ambient air temperature.
    air_temperature: Celsius
    #: :math:`\delta~\left[\text{radian}\right]`. Wind direction east of north.
    wind_direction: Radian
    #: :math:`v~\left[\text{m}~\text{s}^{-1}\right]`. Wind velocity
    wind_speed: MeterPerSecond
    #: :math:`N_s`. The clearness ratio (or clearness number in
    #: :cite:p:`sharma1965interrelationships,cigre207`).
    clearness_ratio: Unitless = 1
