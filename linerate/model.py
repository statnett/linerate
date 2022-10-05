from typing import Dict

from . import equations, solver
from .types import Span, Weather
from .units import Ampere, Celsius, Date, JoulePerKilogramPerKelvin, OhmPerMeter, WattPerMeter

__all__ = ["Cigre601"]


class Cigre601:
    def __init__(self, span: Span, weather: Weather, time: Date):
        self.span = span
        self.weather = weather
        self.time = time

    def compute_resistance(self, conductor_temperature: Celsius, current: Ampere) -> OhmPerMeter:
        resistance = equations.joule_heating.compute_resistance(
            conductor_temperature,
            self.span.conductor.resistance_at_20c,
            self.span.conductor.linear_resistance_coefficient_20c,
            self.span.conductor.quadratic_resistance_coefficient_20c,
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

    def compute_joule_heating(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        resistance = self.compute_resistance(conductor_temperature, current)
        return equations.joule_heating.compute_joule_heating(current, resistance)

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

    def compute_radiative_cooling(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> WattPerMeter:
        return equations.radiative_cooling.compute_radiative_cooling(
            surface_temperature=conductor_temperature,
            air_temperature=self.weather.air_temperature,
            conductor_diameter=self.span.conductor.conductor_diameter,
            conductor_emissivity=self.span.conductor.emissivity,
        )

    def compute_heat_balance(self, conductor_temperature: Celsius, current: Ampere) -> WattPerMeter:
        P_j = self.compute_joule_heating(conductor_temperature, current)
        P_s = self.compute_solar_heating(conductor_temperature, current)
        # TODO: P_s is constant and can be cached
        P_c = self.compute_convective_cooling(conductor_temperature, current)
        P_r = self.compute_radiative_cooling(conductor_temperature, current)
        return P_j + P_s - P_c - P_r

    def compute_steady_state_ampacity(
        self,
        max_conductor_temperature: Celsius,
        min_ampacity: Ampere = 0,
        max_ampacity: Ampere = 5000,
        tolerance: float = 1.0,
    ) -> Ampere:
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
        T = solver.compute_conductor_temperature(
            self.compute_heat_balance,
            current=current,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            tolerance=tolerance,
        )
        n = self.span.num_conductors
        return T / n

    def compute_temperature_gradient(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> Celsius:
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

    def compute_info(
        self, conductor_temperature: Celsius, current: Ampere
    ) -> Dict[str, WattPerMeter]:
        return {
            "convective_cooling": self.compute_convective_cooling(conductor_temperature, current),
            "radiative_cooling": self.compute_radiative_cooling(conductor_temperature, current),
            "joule_heating": self.compute_joule_heating(conductor_temperature, current),
            "solar_heating": self.compute_solar_heating(conductor_temperature, current),
        }
