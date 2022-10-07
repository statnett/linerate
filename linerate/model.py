"""
These thermal model classes provide an easy-to-use interface to compute the thermal rating.
They store conductor and span metadata as well as the weather parameters, compute all the
heating and cooling effects and use those to estimate the thermal rating and conductor
temperature. All numerical heavy-lifting is handled by the ``linerate.equations`` and the
``linerate.solver`` modules.
"""
from abc import ABC, abstractmethod
from typing import Dict

from . import equations, solver
from .types import Span, Weather
from .units import Ampere, Celsius, Date, JoulePerKilogramPerKelvin, OhmPerMeter, WattPerMeter

__all__ = ["ThermalModel", "Cigre601"]


def _copy_docstring(target_function):
    def inner(func):
        func.__doc__ = target_function.__doc__
        return func

    return inner


class ThermalModel(ABC):
    """Abstract class for a minimal conductor thermal model."""

    @abstractmethod
    def __init__(self, span: Span, weather: Weather):
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
        raise NotImplementedError

    @abstractmethod
    def compute_joule_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        r"""Compute the Joule heating, :math:`P_j~\left[\text{W}~\text{m}^{-1}\right]`.

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
        raise NotImplementedError

    @abstractmethod
    def compute_solar_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        r"""Compute the Joule heating, :math:`P_j~\left[\text{W}~\text{m}^{-1}\right]`.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_s~\left[\text{W}~\text{m}^{-1}\right]`. The solar heating.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_convective_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        r"""Compute the convective cooling, :math:`P_j~\left[\text{W}~\text{m}^{-1}\right]`.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_c~\left[\text{W}~\text{m}^{-1}\right]`. The convective cooling.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_radiative_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        r"""Compute the radiative cooling, :math:`P_j~\left[\text{W}~\text{m}^{-1}\right]`.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`P_r~\left[\text{W}~\text{m}^{-1}\right]`. The radiative cooling heating.
        """
        raise NotImplementedError

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
        P_s = self.compute_solar_heating(conductor_temperature, current)
        P_c = self.compute_convective_cooling(conductor_temperature, current)
        P_r = self.compute_radiative_cooling(conductor_temperature, current)
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
        Union[float, float64, ndarray[Any, dtype[float64]]]
            A dictionary with the magnitude of the different heating and cooling effects.
        """
        return {
            "convective_cooling": self.compute_convective_cooling(conductor_temperature, current),
            "radiative_cooling": self.compute_radiative_cooling(conductor_temperature, current),
            "joule_heating": self.compute_joule_heating(conductor_temperature, current),
            "solar_heating": self.compute_solar_heating(conductor_temperature, current),
        }

    def compute_steady_state_ampacity(
        self,
        max_conductor_temperature: Celsius,
        min_ampacity: Ampere = 0,
        max_ampacity: Ampere = 5000,
        tolerance: float = 1.0,
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
        T = solver.compute_conductor_temperature(
            self.compute_heat_balance,
            current=current,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            tolerance=tolerance,
        )
        n = self.span.num_conductors
        return T / n


class Cigre601(ThermalModel):
    def __init__(self, span: Span, weather: Weather, time: Date):
        super().__init__(self.span, self.weather)
        self.time = time

    @_copy_docstring(ThermalModel.compute_resistance)
    def compute_resistance(self, conductor_temperature: Celsius, current: Ampere) -> OhmPerMeter:
        resistance = equations.joule_heating.compute_resistance(
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

        return equations.joule_heating.correct_resistance_acsr_magnetic_core_loss(
            ac_resistance=resistance,
            current=current,
            aluminium_cross_section_area=A,
            constant_magnetic_effect=b,
            current_density_proportional_magnetic_effect=m,
            max_relative_increase=max_increase,
        )

    @_copy_docstring(ThermalModel.compute_resistance)
    def compute_joule_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        resistance = self.compute_resistance(conductor_temperature, current)
        return equations.joule_heating.compute_joule_heating(current, resistance)

    @_copy_docstring(ThermalModel.compute_resistance)
    def compute_solar_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        alpha_s = self.span.conductor.solar_absorptivity
        F = self.span.ground_albedo
        phi = self.span.latitude
        gamma_c = self.span.conductor_azimuth
        y = self.span.conductor_altitude
        N_s = self.weather.clearness_ratio
        D = self.span.conductor.conductor_diameter

        sin_H_s, sin_eta = equations.solar_heating.compute_sin_solar_angles(
            phi, self.span.longitude, self.time, gamma_c
        )

        I_B = equations.solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y)
        I_d = equations.solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)
        I_T = equations.solar_heating.compute_global_radiation_intensity(
            I_B, I_d, F, sin_eta, sin_H_s
        )
        return equations.solar_heating.compute_solar_heating(
            alpha_s,
            I_T,
            D,
        )

    @_copy_docstring(ThermalModel.compute_resistance)
    def compute_convective_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        D = self.span.conductor.conductor_diameter
        d = self.span.conductor.outer_layer_strand_diameter
        y = self.span.conductor_altitude
        beta = self.span.inclination
        V = self.weather.wind_speed
        T_a = self.weather.air_temperature
        T_c = conductor_temperature
        T_f = 0.5 * (T_c + T_a)

        # Compute physical quantities
        lambda_f = equations.convective_cooling.compute_thermal_conductivity_of_air(T_f)
        mu_f = equations.convective_cooling.compute_dynamic_viscosity_of_air(T_f)
        gamma_f = equations.convective_cooling.compute_air_density(T_f, y)
        nu_f = equations.convective_cooling.compute_kinematic_viscosity_of_air(mu_f, gamma_f)
        c_f: JoulePerKilogramPerKelvin = 1005
        delta = equations.math.compute_angle_of_attack(
            self.weather.wind_direction, self.span.conductor_azimuth
        )

        # Compute unitless quantities
        Re = equations.convective_cooling.compute_reynolds_number(V, D, nu_f)
        Gr = equations.convective_cooling.compute_grashof_number(D, T_c, T_a, nu_f)
        Pr = equations.convective_cooling.compute_prandtl_number(lambda_f, mu_f, c_f)
        Rs = equations.convective_cooling.compute_conductor_roughness(D, d)

        # Compute nusselt numbers
        Nu_90 = equations.convective_cooling.compute_perpendicular_flow_nusseltnumber(
            reynolds_number=Re, conductor_roughness=Rs
        )
        Nu_delta = equations.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
            Nu_90, delta, Rs
        )

        Nu_0 = equations.convective_cooling.compute_horizontal_natural_nusselt_number(Gr, Pr)
        Nu_beta = equations.convective_cooling.correct_natural_nusselt_number_inclination(
            Nu_0, beta, Rs
        )

        Nu = equations.convective_cooling.compute_nusselt_number(
            forced_convection_nusselt_number=Nu_delta, natural_nusselt_number=Nu_beta
        )

        return equations.convective_cooling.compute_convective_cooling(
            surface_temperature=conductor_temperature,
            air_temperature=self.weather.air_temperature,
            nusselt_number=Nu,
            thermal_conductivity_of_air=lambda_f,
        )

    @_copy_docstring(ThermalModel.compute_resistance)
    def compute_radiative_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        return equations.radiative_cooling.compute_radiative_cooling(
            surface_temperature=conductor_temperature,
            air_temperature=self.weather.air_temperature,
            conductor_diameter=self.span.conductor.conductor_diameter,
            conductor_emissivity=self.span.conductor.emissivity,
        )

    def compute_temperature_gradient(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> Celsius:
        r"""Estimate the difference between the core temperature and the surface temperature.

        Parameters
        ----------
        conductor_temperature:
            :math:`T_\text{av}~\left[^\circ\text{C}\right]`. The average conductor temperature.
        current:
            :math:`I~\left[\text{A}\right]`. The current.

        Returns
        -------
        Union[float, float64, ndarray[Any, dtype[float64]]]
            :math:`T_c - T_s~\left[^\circ \text{C}\right]`. The difference between the core and the
            surface temperature of the conductor.
        """
        n = self.span.num_conductors
        T_c = conductor_temperature
        I = current / n  # noqa
        R = self.compute_resistance(conductor_temperature=T_c, current=I)
        return equations.convective_cooling.compute_temperature_gradient(
            total_heat_gain=I * R,
            conductor_thermal_conductivity=self.span.conductor.thermal_conductivity,  # type: ignore  # noqa
            core_diameter=self.span.conductor.core_diameter,
            conductor_diameter=self.span.conductor.conductor_diameter,
        )
