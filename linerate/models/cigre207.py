from linerate.equations import (
    cigre207,
    convective_cooling,
    dimensionless,
    math,
    solar_angles,
    solar_heating,
)
from linerate.equations.math import switch_cos_sin
from linerate.models.thermal_model import ThermalModel, _copy_method_docstring
from linerate.types import Span, Weather
from linerate.units import Ampere, Celsius, Date, OhmPerMeter, WattPerMeter


class Cigre207(ThermalModel):
    def __init__(
        self,
        span: Span,
        weather: Weather,
        time: Date,
        include_diffuse_radiation: bool = True,
        direct_radiation_factor: float = 1.0,
    ):
        super().__init__(span, weather)
        self.time = time
        self.include_diffuse_radiation = include_diffuse_radiation
        self.direct_radiation_factor = direct_radiation_factor

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
        phi = self.span.latitude
        gamma_c = self.span.conductor_azimuth
        y = self.span.conductor_altitude
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

        I_B = self.direct_radiation_factor * cigre207.solar_heating.compute_direct_solar_radiation(
            sin_H_s, y
        )
        if self.include_diffuse_radiation:
            I_d = cigre207.solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)
            F = self.span.ground_albedo
        else:
            I_d = 0
            F = 0
        I_T = cigre207.solar_heating.compute_global_radiation_intensity(
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
        V = self.weather.wind_speed
        y = self.span.conductor_altitude
        T_a = self.weather.air_temperature
        T_c = conductor_temperature
        T_f = 0.5 * (T_c + T_a)

        # Compute physical quantities
        lambda_f = cigre207.convective_cooling.compute_thermal_conductivity_of_air(T_f)
        nu_f = cigre207.convective_cooling.compute_kinematic_viscosity_of_air(T_f)
        delta = math.compute_angle_of_attack(
            self.weather.wind_direction, self.span.conductor_azimuth
        )

        # Compute unitless quantities
        rho_r = cigre207.convective_cooling.compute_relative_air_density(y)
        Re = cigre207.convective_cooling.compute_reynolds_number(V, D, nu_f, rho_r)
        Gr = dimensionless.compute_grashof_number(D, T_c, T_a, nu_f)
        Pr = cigre207.convective_cooling.compute_prandtl_number(T_f)
        Rs = dimensionless.compute_conductor_roughness(D, d)

        # Compute nusselt numbers
        Nu_90 = cigre207.convective_cooling.compute_perpendicular_flow_nusseltnumber(
            reynolds_number=Re, conductor_roughness=Rs
        )
        Nu_delta = cigre207.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
            Nu_90, delta
        )
        Nu_cor = cigre207.convective_cooling.compute_low_wind_speed_nusseltnumber(Nu_90)

        Nu_0 = cigre207.convective_cooling.compute_horizontal_natural_nusselt_number(Gr, Pr)

        Nu = cigre207.convective_cooling.compute_nusselt_number(
            forced_convection_nusselt_number=Nu_delta,
            natural_nusselt_number=Nu_0,
            low_wind_nusselt_number=Nu_cor,
            wind_speed=V,
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

    @_copy_method_docstring(ThermalModel)
    def compute_resistance(self, conductor_temperature: Celsius, current: Ampere) -> OhmPerMeter:
        return super().compute_resistance(
            conductor_temperature=conductor_temperature, current=current
        )
