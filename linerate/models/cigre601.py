from numbers import Real

import numpy as np

from linerate.equations import (
    cigre601,
    convective_cooling,
    dimensionless,
    math,
    solar_angles,
    solar_heating,
)
from linerate.equations.math import switch_cos_sin
from linerate.models.thermal_model import ThermalModel, _copy_method_docstring
from linerate.types import Span, Weather
from linerate.units import (
    Ampere,
    Celsius,
    Date,
    JoulePerKilogramPerKelvin,
    OhmPerMeter,
    WattPerMeter,
)


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

        omega = solar_angles.compute_hour_angle_relative_to_noon(self.time, self.span.longitude)
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
        return solar_heating.compute_solar_heating(
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
            dimensionless.compute_reynolds_number(V, D, nu_f),
            self.max_reynolds_number,
        )
        Gr = dimensionless.compute_grashof_number(D, T_c, T_a, nu_f)
        Pr = dimensionless.compute_prandtl_number(lambda_f, mu_f, c_f)
        Rs = dimensionless.compute_conductor_roughness(D, d)

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

        return convective_cooling.compute_convective_cooling(
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
