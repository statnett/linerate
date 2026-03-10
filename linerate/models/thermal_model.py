from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from functools import partial
from linerate import solver
from linerate.equations import joule_heating, radiative_cooling, heat_capacity
from linerate.types import BaseWeather, Span, conductor_heat_capacity_defined
from linerate.units import Ampere, Celsius, Duration, OhmPerMeter, WattPerMeter


def _copy_method_docstring(parent_class):
    def inner(func):
        func.__doc__ = getattr(parent_class, func.__name__).__doc__
        return func

    return inner


class ThermalModel(ABC):
    """Abstract class for a minimal conductor thermal model."""

    @abstractmethod
    def __init__(self, span: Span, weather: BaseWeather):
        self.span = span
        self.weather = weather

    @abstractmethod
    def compute_resistance(self, conductor_temperature: Celsius, current: Ampere) -> OhmPerMeter:
        r"""Compute the conductor resistance, :math:`R~\left[\Omega~\text{m}^{-1}\right]`.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`R~\left[\Omega\right]`. The resistance at the given temperature and current.
        """
        resistance = joule_heating.compute_resistance(
            conductor_temperature,
            temperature1=self.span.conductor.temperature1,
            temperature2=self.span.conductor.temperature2,
            resistance_at_temperature1=self.span.conductor.resistance_at_temperature1,
            resistance_at_temperature2=self.span.conductor.resistance_at_temperature2,
        )

        A = self.span.conductor.aluminium_cross_section_area
        b = self.span.conductor.constant_magnetic_effect
        m = self.span.conductor.current_density_proportional_magnetic_effect
        max_increase = self.span.conductor.max_magnetic_core_relative_resistance_increase

        return joule_heating.correct_resistance_acsr_magnetic_core_loss(
            ac_resistance=resistance,
            current=current,
            aluminium_cross_section_area=A,
            constant_magnetic_effect=b,
            current_density_proportional_magnetic_effect=m,
            max_relative_increase=max_increase,
        )

    @abstractmethod
    def compute_joule_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        r"""Compute the Joule heating, :math:`P_J~\left[\text{W}~\text{m}^{-1}\right]`.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_J~\left[\text{W}~\text{m}^{-1}\right]`. The Joule heating.
        """
        resistance = self.compute_resistance(conductor_temperature, current)
        return joule_heating.compute_joule_heating(current, resistance)

    @abstractmethod
    def compute_solar_heating(
        self,
    ) -> WattPerMeter:
        r"""Compute the solar heating, :math:`P_S~\left[\text{W}~\text{m}^{-1}\right]`.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_s~\left[\text{W}~\text{m}^{-1}\right]`. The solar heating.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_convective_cooling(
        self,
        conductor_temperature: Celsius,
    ) -> WattPerMeter:
        r"""Compute the convective cooling, :math:`P_c~\left[\text{W}~\text{m}^{-1}\right]`.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_c~\left[\text{W}~\text{m}^{-1}\right]`. The convective cooling.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_radiative_cooling(
        self,
        conductor_temperature: Celsius,
    ) -> WattPerMeter:
        r"""Compute the radiative cooling, :math:`P_r~\left[\text{W}~\text{m}^{-1}\right]`.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_r~\left[\text{W}~\text{m}^{-1}\right]`. The radiative cooling heating.
        """
        return radiative_cooling.compute_radiative_cooling(
            surface_temperature=conductor_temperature,
            air_temperature=self.weather.air_temperature,
            conductor_diameter=self.span.conductor.conductor_diameter,
            conductor_emissivity=self.span.conductor.emissivity,
        )

    def compute_heat_balance(self, conductor_temperature: Celsius, current: Ampere) -> WattPerMeter:
        r"""Compute the conductor's heat balance. Positive means heating and negative means cooling.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_J + P_s - P_c - P_r~\left[\text{W}~\text{m}^{-1}\right]`. The heat balance.
        """
        P_j = self.compute_joule_heating(conductor_temperature, current)
        P_s = self.compute_solar_heating()
        P_c = self.compute_convective_cooling(conductor_temperature)
        P_r = self.compute_radiative_cooling(conductor_temperature)
        return P_j + P_s - P_c - P_r

    def compute_info(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> Dict[str, WattPerMeter]:
        r"""Create a dictionary with the different heating and cooling effects.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Dict[str, WattPerMeter]
            A dictionary with the magnitude of the different heating and cooling effects.
        """
        return {
            "convective_cooling": self.compute_convective_cooling(conductor_temperature),
            "radiative_cooling": self.compute_radiative_cooling(conductor_temperature),
            "joule_heating": self.compute_joule_heating(conductor_temperature, current),
            "solar_heating": self.compute_solar_heating(),
        }

    def compute_steady_state_ampacity(
        self,
        max_conductor_temperature: Celsius,
        min_ampacity: Ampere = 0,
        max_ampacity: Ampere = 5000,
        tolerance: float = 1.0,
        accept_invalid_values: bool = False,
    ) -> Ampere:
        r"""Use the bisection method to compute the steady-state thermal rating (ampacity).

        Parameters
        ----------
        max_conductor_temperature:
            :math:`T_\text{max}~\left[^\circ\text{C}\right]`. Maximum allowed conductor temperature
        min_ampacity:
            :math:`I_\text{min}~\left[\text{A}\right]`. Lower bound for the numerical scheme for
            computing the ampacity
        max_ampacity:
            :math:`I_\text{min}~\left[\text{A}\right]`. Upper bound for the numerical scheme for
            computing the ampacity
        tolerance:
            :math:`\Delta I~\left[\text{A}\right]`. The numerical accuracy of the ampacity. The
            bisection iterations will stop once the numerical ampacity uncertainty is below
            :math:`\Delta I`. The bisection method will run for
            :math:`\left\lceil\frac{I_\text{min} - I_\text{min}}{\Delta I}\right\rceil` iterations.
        accept_invalid_values:
            If True, np.nan is returned whenever the current cannot be found within the provided
            search interval. If False, a ValueError will be raised instead.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`I~\left[\text{A}\right]`. The thermal rating.
        """
        I = solver.compute_conductor_ampacity(  # noqa
            self.compute_heat_balance,
            max_conductor_temperature=max_conductor_temperature,
            min_ampacity=min_ampacity,
            max_ampacity=max_ampacity,
            tolerance=tolerance,
            accept_invalid_values=accept_invalid_values,
        )
        n = self.span.num_conductors
        return I * n

    def compute_conductor_temperature(
        self,
        current: Ampere,
        min_temperature: Celsius = -30,
        max_temperature: Celsius = 150,
        tolerance: float = 0.5,
    ) -> Celsius:
        r"""Use the bisection method to compute the steady state conductor temperature.

        Parameters
        ----------
        current:
            :math:`I_\text{max}~\left[\text{A}\right]`. The current flowing through the conductor.
            NOTE that the current is the total current for all conductors in the span. When
            computing the temperature, the current is divided by the number of conductors.
        min_temperature:
            :math:`T_\text{min}~\left[^\circ\text{C}\right]`. Lower bound for the numerical scheme
            for computing the temperature
        max_temperature:
            :math:`T_\text{max}~\left[^\circ\text{C}\right]`. Upper bound for the numerical scheme
            for computing the temperature
        tolerance:
            :math:`\Delta T~\left[^\circ\text{C}\right]`. The numerical accuracy of the
            temperature. The bisection iterations will stop once the numerical temperature
            uncertainty is below :math:`\Delta T`. The bisection method will run for
            :math:`\left\lceil\frac{T_\text{min} - T_\text{min}}{\Delta T}\right\rceil` iterations.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`I~\left[\text{A}\right]`. The thermal rating.
        """
        n = self.span.num_conductors
        T = solver.compute_conductor_temperature(
            self.compute_heat_balance,
            current=current / n,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            tolerance=tolerance,
        )
        return T

    def compute_heat_capacity_per_unit_length(self, conductor_temperature: Celsius):
        conductor = self.span.conductor
        if conductor_heat_capacity_defined(conductor):
            mc_s = heat_capacity.calculate_heat_capacity_per_unit_length(
                mass_per_unit_length=conductor.steel_mass_per_unit_length,
                specific_heat_capacity_at_20_celsius=conductor.steel_specific_heat_capacity_at_20_celsius,
                specific_heat_capacity_coefficient=conductor.steel_specific_heat_capacity_temperature_coefficient,
                conductor_temperature=conductor_temperature,
            )
            mc_a = heat_capacity.calculate_heat_capacity_per_unit_length(
                mass_per_unit_length=conductor.aluminum_mass_per_unit_length,
                specific_heat_capacity_at_20_celsius=conductor.aluminum_specific_heat_capacity_at_20_celsius,
                specific_heat_capacity_coefficient=conductor.aluminum_specific_heat_capacity_temperature_coefficient,
                conductor_temperature=conductor_temperature,
            )
            return mc_s + mc_a
        else:
            raise RuntimeError("Heat capacity data must be defined to calculate heat capacity.")

    def compute_final_temperature(
        self,
        initial_conductor_temperature: Celsius,
        heating_time: Duration,
        current: Ampere,
        time_step: Duration = np.timedelta64(60, "s"),
    ) -> Ampere:
        time_step = np.broadcast_to(time_step, heating_time.shape)
        step_count = heating_time // time_step
        remainder = heating_time % time_step
        dt = time_step / np.timedelta64(1, "s")
        modification_mask = step_count > 0
        temperature = initial_conductor_temperature
        while np.any(modification_mask):
            heat_capacity_ = self.compute_heat_capacity_per_unit_length(temperature)
            heat_balance = self.compute_heat_balance(temperature, current=current)
            dT = (heat_balance / heat_capacity_) * dt
            temperature = temperature + modification_mask * dT
            step_count -= 1
            modification_mask = step_count > 0
        if np.any(remainder > 0):
            heat_capacity_ = self.compute_heat_capacity_per_unit_length(temperature)
            heat_balance = self.compute_heat_balance(temperature, current=current)
            dT = (heat_balance / heat_capacity_) * remainder
            temperature = temperature + dT
        return temperature

    def compute_temporary_ampacity(
        self,
        max_conductor_temperature: Celsius,
        heating_time: Duration,
        initial_conductor_temperature: Celsius,
        time_step: Duration = np.timedelta64(60, "s"),
        min_ampacity: Ampere = 0,
        max_ampacity: Ampere = 5000,
        tolerance: float = 1.0,
        accept_invalid_values: bool = False,
    ):
        n = self.span.num_conductors
        I = solver.compute_conductor_transient_ampacity(  # noqa
            partial(self.compute_final_temperature, time_step=time_step),
            max_conductor_temperature,
            initial_conductor_temperature,
            heating_time,
            min_ampacity,
            max_ampacity,
            tolerance,
            accept_invalid_values,
        )
        return I * n
