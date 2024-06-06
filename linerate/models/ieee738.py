from numbers import Real

import numpy as np

from linerate.equations import dimensionless, ieee738, math, solar_angles
from linerate.models.thermal_model import ThermalModel, _copy_method_docstring
from linerate.types import Span, Weather
from linerate.units import Ampere, Celsius, Date, OhmPerMeter, WattPerMeter


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

        omega = solar_angles.compute_hour_angle_relative_to_noon(self.time, self.span.longitude)
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
            dimensionless.compute_reynolds_number(V, D, nu_f),  # N_Re in IEEE
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
