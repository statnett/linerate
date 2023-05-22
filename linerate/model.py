"""
These thermal model classes provide an easy-to-use interface to compute the thermal rating.
They store conductor and span metadata as well as the weather parameters, compute all the
heating and cooling effects and use those to estimate the thermal rating and conductor
temperature. All numerical heavy-lifting is handled by the ``linerate.equations`` and the
``linerate.solver`` modules.
"""
from abc import ABC, abstractmethod
from numbers import Real
from typing import Dict

import numpy as np

from linerate import solver
from linerate.equations import (
    cigre601,
    ieee738,
    joule_heating,
    math,
    radiative_cooling,
    solar_angles,
)
from linerate.equations.math import switch_cos_sin
from linerate.types import Span, Weather
from linerate.units import (
    Ampere,
    Celsius,
    Date,
    JoulePerKilogramPerKelvin,
    OhmPerMeter,
    WattPerMeter,
)

__all__ = ["ThermalModel", "Cigre601", "IEEE738"]


def _copy_method_docstring(parent_class):
    def inner(func):
        func.__doc__ = getattr(parent_class, func.__name__).__doc__
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
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        r"""Compute the solar heating, :math:`P_S~\left[\text{W}~\text{m}^{-1}\right]`.

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
        r"""Compute the convective cooling, :math:`P_c~\left[\text{W}~\text{m}^{-1}\right]`.

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
        r"""Compute the radiative cooling, :math:`P_r~\left[\text{W}~\text{m}^{-1}\right]`.

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
        Dict[str, WattPerMeter]
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
    def __init__(
        self,
        span: Span,
        weather: Weather,
        time: Date,
        max_reynolds_number: Real = 4000,  # Max value of the angle correction in CIGRE601
    ):
        super().__init__(span, weather)
        self.time = time
        self.max_reynolds_number = max_reynolds_number

    @_copy_method_docstring(ThermalModel)
    def compute_resistance(self, conductor_temperature: Celsius, current: Ampere) -> OhmPerMeter:
        return super().compute_resistance(
            conductor_temperature=conductor_temperature, current=current
        )

    @_copy_method_docstring(ThermalModel)
    def compute_joule_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        return super().compute_joule_heating(
            conductor_temperature=conductor_temperature, current=current
        )

    @_copy_method_docstring(ThermalModel)
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

        omega = solar_angles.compute_hour_angle_relative_to_noon(self.time)
        delta = solar_angles.compute_solar_declination(self.time)
        sin_H_s = solar_angles.compute_sin_solar_altitude(phi, delta, omega)
        chi = solar_angles.compute_solar_azimuth_variable(phi, delta, omega)
        C = solar_angles.compute_solar_azimuth_constant(chi, omega)
        gamma_s = solar_angles.compute_solar_azimuth(C, chi)  # Z_c in IEEE
        cos_eta = solar_angles.compute_cos_solar_effective_incidence_angle(
            sin_H_s, gamma_s, gamma_c
        )
        sin_eta = switch_cos_sin(cos_eta)

        I_B = cigre601.solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y)
        I_d = cigre601.solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)
        I_T = cigre601.solar_heating.compute_global_radiation_intensity(
            I_B, I_d, F, sin_eta, sin_H_s
        )
        return cigre601.solar_heating.compute_solar_heating(
            alpha_s,
            I_T,
            D,
        )

    @_copy_method_docstring(ThermalModel)
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
        lambda_f = cigre601.convective_cooling.compute_thermal_conductivity_of_air(T_f)
        mu_f = cigre601.convective_cooling.compute_dynamic_viscosity_of_air(T_f)
        gamma_f = cigre601.convective_cooling.compute_air_density(T_f, y)
        nu_f = cigre601.convective_cooling.compute_kinematic_viscosity_of_air(mu_f, gamma_f)
        c_f: JoulePerKilogramPerKelvin = 1005
        delta = math.compute_angle_of_attack(
            self.weather.wind_direction, self.span.conductor_azimuth
        )

        # Compute unitless quantities
        Re = np.minimum(
            cigre601.convective_cooling.compute_reynolds_number(V, D, nu_f),
            self.max_reynolds_number,
        )
        Gr = cigre601.convective_cooling.compute_grashof_number(D, T_c, T_a, nu_f)
        Pr = cigre601.convective_cooling.compute_prandtl_number(lambda_f, mu_f, c_f)
        Rs = cigre601.convective_cooling.compute_conductor_roughness(D, d)

        # Compute nusselt numbers
        Nu_90 = cigre601.convective_cooling.compute_perpendicular_flow_nusseltnumber(
            reynolds_number=Re, conductor_roughness=Rs
        )
        Nu_delta = cigre601.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
            Nu_90, delta, Rs
        )

        Nu_0 = cigre601.convective_cooling.compute_horizontal_natural_nusselt_number(Gr, Pr)
        Nu_beta = cigre601.convective_cooling.correct_natural_nusselt_number_inclination(
            Nu_0, beta, Rs
        )

        Nu = cigre601.convective_cooling.compute_nusselt_number(
            forced_convection_nusselt_number=Nu_delta, natural_nusselt_number=Nu_beta
        )

        return cigre601.convective_cooling.compute_convective_cooling(
            surface_temperature=conductor_temperature,
            air_temperature=self.weather.air_temperature,
            nusselt_number=Nu,
            thermal_conductivity_of_air=lambda_f,
        )

    @_copy_method_docstring(ThermalModel)
    def compute_radiative_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        return super().compute_radiative_cooling(
            conductor_temperature=conductor_temperature, current=current
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
        return cigre601.convective_cooling.compute_temperature_gradient(
            total_heat_gain=I * R,
            conductor_thermal_conductivity=self.span.conductor.thermal_conductivity,  # type: ignore  # noqa
            core_diameter=self.span.conductor.core_diameter,
            conductor_diameter=self.span.conductor.conductor_diameter,
        )


class IEEE738(ThermalModel):
    def __init__(
        self,
        span: Span,
        weather: Weather,
        time: Date,
        max_reynolds_number: Real = 50_000,  # Max Reynolds number for forced convection
    ):
        super().__init__(span, weather)
        self.time = time
        self.max_reynolds_number = max_reynolds_number

    @_copy_method_docstring(ThermalModel)
    def compute_resistance(self, conductor_temperature: Celsius, current: Ampere) -> OhmPerMeter:
        return super().compute_resistance(
            conductor_temperature=conductor_temperature, current=current
        )

    @_copy_method_docstring(ThermalModel)
    def compute_joule_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        return super().compute_joule_heating(
            conductor_temperature=conductor_temperature, current=current
        )

    @_copy_method_docstring(ThermalModel)
    def compute_solar_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        alpha_s = self.span.conductor.solar_absorptivity  # alpha in IEEE
        phi = self.span.latitude  # Lat in IEEE
        gamma_c = self.span.conductor_azimuth  # Z_l i IEEE
        y = self.span.conductor_altitude  # H_e in IEEE
        D = self.span.conductor.conductor_diameter  # D_0 in IEEE

        omega = solar_angles.compute_hour_angle_relative_to_noon(self.time)
        delta = solar_angles.compute_solar_declination(self.time)
        sin_H_c = solar_angles.compute_sin_solar_altitude(phi, delta, omega)
        Q_s = ieee738.solar_heating.compute_total_heat_flux_density(sin_H_c, True)
        K_solar = ieee738.solar_heating.compute_solar_altitude_correction_factor(y)
        Q_se = ieee738.solar_heating.compute_elevation_correction_factor(K_solar, Q_s)
        chi = solar_angles.compute_solar_azimuth_variable(phi, delta, omega)
        C = solar_angles.compute_solar_azimuth_constant(chi, omega)
        Z_c = solar_angles.compute_solar_azimuth(C, chi)
        cos_theta = solar_angles.compute_cos_solar_effective_incidence_angle(sin_H_c, Z_c, gamma_c)

        return ieee738.solar_heating.compute_solar_heating(alpha_s, Q_se, cos_theta, D)

    @_copy_method_docstring(ThermalModel)
    def compute_convective_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        D = self.span.conductor.conductor_diameter  # D_0 in IEEE
        y = self.span.conductor_altitude  # H_e in IEEE
        V = self.weather.wind_speed  # V_w in IEEE
        T_a = self.weather.air_temperature
        T_c = conductor_temperature
        T_f = 0.5 * (T_c + T_a)  # T_film in IEEE

        mu_f = ieee738.convective_cooling.compute_dynamic_viscosity_of_air(T_f)
        rho_f = ieee738.convective_cooling.compute_air_density(T_f, y)
        nu_f = ieee738.convective_cooling.compute_kinematic_viscosity_of_air(mu_f, rho_f)
        Re = np.minimum(
            ieee738.convective_cooling.compute_reynolds_number(V, D, nu_f),  # N_Re in IEEE
            self.max_reynolds_number,
        )
        delta = math.compute_angle_of_attack(
            self.weather.wind_direction, self.span.conductor_azimuth
        )  # Phi in IEEE
        K_angle = ieee738.convective_cooling.compute_wind_direction_factor(delta)
        k_f = ieee738.convective_cooling.compute_thermal_conductivity_of_air(T_f)
        q_cf = ieee738.convective_cooling.compute_forced_convection(K_angle, Re, k_f, T_c, T_a)
        q_cn = ieee738.convective_cooling.compute_natural_convection(rho_f, D, T_c, T_a)
        return ieee738.convective_cooling.compute_convective_cooling(q_cf, q_cn)

    @_copy_method_docstring(ThermalModel)
    def compute_radiative_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        return super().compute_radiative_cooling(
            conductor_temperature=conductor_temperature, current=current
        )
