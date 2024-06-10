import numpy as np
from pytest import approx

from linerate.equations import cigre207, dimensionless


def test_matches_example1():
    # See Appendix 1, Example 1 in Cigre 207
    y = 1600
    rho_r = cigre207.convective_cooling.compute_relative_air_density(y)
    assert rho_r == approx(0.8306, rel=1e-4)
    T_s = 57
    T_amb = 40
    T_f = (T_s + T_amb) / 2
    nu_f = cigre207.convective_cooling.compute_kinematic_viscosity_of_air(T_f)
    assert nu_f == approx(1.78e-5, rel=1e-3)
    v = 2
    D = 0.0286
    Re = cigre207.convective_cooling.compute_reynolds_number(v, D, nu_f, rho_r)
    assert Re == approx(2670, rel=1e-3)
    lambda_f = cigre207.convective_cooling.compute_thermal_conductivity_of_air(T_f)
    assert lambda_f == approx(0.0277, 1e-3)


def test_matches_example1_nusselt_number():
    # See Appendix 1, Example 1 in Cigre 207
    D = 0.0286
    d = 0.00318
    Rs = dimensionless.compute_conductor_roughness(D, d)
    Re = 2670
    Nu_90 = cigre207.convective_cooling.compute_perpendicular_flow_nusseltnumber(Re, Rs)
    assert Nu_90 == approx(26.45, 1e-4)
    Nu_45 = cigre207.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, np.radians(45)
    )
    assert Nu_45 == approx(22.34, rel=1e-4)


def test_matches_example2():
    # See Appendix 1, Example 2 in Cigre 207
    y = 1600
    rho_r = cigre207.convective_cooling.compute_relative_air_density(y)
    T_s = 93
    T_amb = 40
    v = 0.2
    D = 0.0286
    T_f = (T_s + T_amb) / 2
    nu_f = cigre207.convective_cooling.compute_kinematic_viscosity_of_air(T_f)
    nu_f_exp = 1.95e-5
    assert nu_f == approx(nu_f_exp, rel=1e-3)
    Re = cigre207.convective_cooling.compute_reynolds_number(v, D, nu_f, rho_r)
    assert Re == approx(243.8, rel=2e-3)
    lambda_f = cigre207.convective_cooling.compute_thermal_conductivity_of_air(T_f)
    assert lambda_f == approx(0.0290, 1e-3)
    d = 0.00318
    Rs = dimensionless.compute_conductor_roughness(D, d)
    Nu_90 = cigre207.convective_cooling.compute_perpendicular_flow_nusseltnumber(Re, Rs)
    assert Nu_90 == approx(8.53, 1e-3)
    Nu_45 = cigre207.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, np.radians(45)
    )
    assert Nu_45 == approx(7.20, rel=2e-3)
    Nu_cor = cigre207.convective_cooling.compute_low_wind_speed_nusseltnumber(Nu_90)
    assert Nu_cor == approx(4.69, rel=1e-3)
    Gr = dimensionless.compute_grashof_number(D, T_s, T_amb, nu_f_exp)
    assert Gr == approx(94387, rel=3e-3)
    Pr = cigre207.convective_cooling.compute_prandtl_number(T_f)
    assert Pr == approx(0.698, rel=1e-3)
    Nu_natural = cigre207.convective_cooling.compute_horizontal_natural_nusselt_number(Gr, Pr)
    assert Nu_natural == approx(7.69, rel=1e-3)
    Nu_eff = cigre207.convective_cooling.compute_nusselt_number(Nu_45, Nu_natural, Nu_cor, v)
    assert Nu_eff == approx(7.69, rel=1e-3)


def test_matches_example3():
    # See Appendix 1, Example 3 in Cigre 207
    y = 1600
    rho_r = cigre207.convective_cooling.compute_relative_air_density(y)
    T_s = 75
    T_amb = 40
    v = 2
    D = 0.0286
    d = 0.00318
    T_f = (T_s + T_amb) / 2
    nu_f = cigre207.convective_cooling.compute_kinematic_viscosity_of_air(T_f)
    nu_f_exp = 1.866e-5
    assert nu_f == approx(nu_f_exp, rel=1e-3)
    Re = cigre207.convective_cooling.compute_reynolds_number(v, D, nu_f, rho_r)
    assert Re == approx(2547.5, rel=1e-3)
    lambda_f = cigre207.convective_cooling.compute_thermal_conductivity_of_air(T_f)
    assert lambda_f == approx(0.0283, 2e-3)
    Rs = dimensionless.compute_conductor_roughness(D, d)
    Nu_90 = cigre207.convective_cooling.compute_perpendicular_flow_nusseltnumber(Re, Rs)
    assert Nu_90 == approx(25.77, 1e-3)
    Nu_45 = cigre207.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, np.radians(45)
    )
    assert Nu_45 == approx(21.8, rel=2e-3)


def test_matches_example4():
    # See Appendix 1, Example 4 in Cigre 207
    y = 1600
    rho_r = cigre207.convective_cooling.compute_relative_air_density(y)
    T_s = 75
    T_amb = 40
    v = 0.4
    D = 0.0286
    T_f = (T_s + T_amb) / 2
    nu_f = cigre207.convective_cooling.compute_kinematic_viscosity_of_air(T_f)
    nu_f_exp = 1.866e-5
    assert nu_f == approx(nu_f_exp, rel=1e-3)
    Re = cigre207.convective_cooling.compute_reynolds_number(v, D, nu_f, rho_r)
    assert Re == approx(509.6, rel=2e-3)
    lambda_f = cigre207.convective_cooling.compute_thermal_conductivity_of_air(T_f)
    assert lambda_f == approx(0.0283, 2e-3)
    d = 0.00318
    Rs = dimensionless.compute_conductor_roughness(D, d)
    Nu_90 = cigre207.convective_cooling.compute_perpendicular_flow_nusseltnumber(Re, Rs)
    assert Nu_90 == approx(12.08, 1e-3)
    Nu_45 = cigre207.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, np.radians(45)
    )
    assert Nu_45 == approx(10.2, rel=2e-3)
    Nu_cor = cigre207.convective_cooling.compute_low_wind_speed_nusseltnumber(Nu_90)
    assert Nu_cor == approx(6.64, rel=1e-3)
    Gr = dimensionless.compute_grashof_number(D, T_s, T_amb, nu_f_exp)
    assert Gr == approx(69922.7, rel=3e-3)
    Pr = cigre207.convective_cooling.compute_prandtl_number(T_f)
    assert Pr == approx(0.701, rel=1e-3)
    Nu_natural = cigre207.convective_cooling.compute_horizontal_natural_nusselt_number(Gr, Pr)
    assert Nu_natural == approx(7.14, rel=1e-3)
    Nu_eff = cigre207.convective_cooling.compute_nusselt_number(Nu_45, Nu_natural, Nu_cor, v)
    assert Nu_eff == approx(10.2, rel=1e-3)


def test_matches_example5():
    # See Appendix 1, Example 5 in Cigre 207
    y = 300
    rho_r = cigre207.convective_cooling.compute_relative_air_density(y)
    assert rho_r == approx(0.966, rel=1e-3)
    T_s = 57
    T_amb = 40
    v = 2
    D = 0.0286
    T_f = (T_s + T_amb) / 2
    nu_f = cigre207.convective_cooling.compute_kinematic_viscosity_of_air(T_f)
    nu_f_exp = 1.78e-5
    assert nu_f == approx(nu_f_exp, rel=1e-3)
    Re = cigre207.convective_cooling.compute_reynolds_number(v, D, nu_f, rho_r)
    assert Re == approx(3106, rel=2e-3)
    lambda_f = cigre207.convective_cooling.compute_thermal_conductivity_of_air(T_f)
    assert lambda_f == approx(0.0277, 2e-3)
    d = 0.00318
    Rs = dimensionless.compute_conductor_roughness(D, d)
    Nu_90 = cigre207.convective_cooling.compute_perpendicular_flow_nusseltnumber(Re, Rs)
    assert Nu_90 == approx(29.85, 1e-3)
    Nu_45 = cigre207.convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, np.radians(45)
    )
    assert Nu_45 == approx(25.21, rel=2e-3)
