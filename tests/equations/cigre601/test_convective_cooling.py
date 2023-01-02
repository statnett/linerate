import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from pytest import approx
from scipy.interpolate import lagrange

import linerate.equations.cigre601.convective_cooling as convective_cooling

# Tests for physical quantities
###############################


@hypothesis.given(total_heat_gain=st.floats(allow_nan=False))
def test_temperature_gradient_scales_linearly_with_heating(total_heat_gain):
    P_T = total_heat_gain
    lambda_ = 0.5 / np.pi  # 2 pi lambda = 1
    D = 2
    D_c = 0
    delta_T = convective_cooling.compute_temperature_gradient(
        total_heat_gain=P_T,
        conductor_thermal_conductivity=lambda_,
        core_diameter=D_c,
        conductor_diameter=D,
    )
    assert delta_T == approx(0.5 * P_T)

    D_c = 1
    delta_T = convective_cooling.compute_temperature_gradient(
        total_heat_gain=P_T,
        conductor_thermal_conductivity=lambda_,
        core_diameter=D_c,
        conductor_diameter=D,
    )
    assert delta_T == approx(
        P_T * (0.5 - np.log(2) / 3)
    )  # 0.5**2 / (1**2 - 0.5**2) = 0.25/0.75 / 1/3


@hypothesis.given(conductor_thermal_conductivity=st.floats(allow_nan=False, min_value=1e-5))
def test_temperature_gradient_scales_inversely_with_heat_conductivity(
    conductor_thermal_conductivity,
):
    P_T = 2 * np.pi
    lambda_ = conductor_thermal_conductivity
    D = 2
    D_c = 0
    delta_T = convective_cooling.compute_temperature_gradient(
        total_heat_gain=P_T,
        conductor_thermal_conductivity=lambda_,
        core_diameter=D_c,
        conductor_diameter=D,
    )
    assert delta_T == approx(0.5 / lambda_)

    D_c = 1
    delta_T = convective_cooling.compute_temperature_gradient(
        total_heat_gain=P_T,
        conductor_thermal_conductivity=lambda_,
        core_diameter=D_c,
        conductor_diameter=D,
    )
    assert delta_T == approx(
        (0.5 - np.log(2) / 3) / lambda_
    )  # 0.5**2 / (1**2 - 0.5**2) = 0.25/0.75 / 1/3


@hypothesis.given(conductor_diameter=st.floats(allow_nan=False, min_value=1e-5, max_value=1e5))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
def test_temperature_gradient_scales_correctly_with_diameter(random_seed, conductor_diameter):
    rng = np.random.default_rng(random_seed)
    P_T = 2 * np.pi
    lambda_ = 1
    D = conductor_diameter
    D_c = 0
    delta_T = convective_cooling.compute_temperature_gradient(
        total_heat_gain=P_T,
        conductor_thermal_conductivity=lambda_,
        core_diameter=D_c,
        conductor_diameter=D,
    )

    assert delta_T == approx(0.5)

    D_c = rng.uniform(0, D * 0.999)
    delta_T = convective_cooling.compute_temperature_gradient(
        total_heat_gain=P_T,
        conductor_thermal_conductivity=lambda_,
        core_diameter=D_c,
        conductor_diameter=D,
    )

    R_scaling = np.log(D / D_c) * (D_c**2) / (D**2 - D_c**2)
    assert delta_T == approx(0.5 - R_scaling)


@pytest.mark.parametrize(
    "total_heat_gain,conductor_thermal_conductivity,core_diameter,conductor_diameter,temperature_difference",  # noqa
    [
        (2, 0.5, 0, 0.5, 1 / np.pi),
        (2, 0.5, 0.1, 0.5, (2 / np.pi) * (0.5 - np.log(5) * 0.01 / 0.24)),
    ],
)
def test_temperature_gradient_with_examples(
    total_heat_gain,
    conductor_thermal_conductivity,
    core_diameter,
    conductor_diameter,
    temperature_difference,
):
    P_T = total_heat_gain
    lambda_ = conductor_thermal_conductivity
    D = conductor_diameter
    D_c = core_diameter
    delta_T = convective_cooling.compute_temperature_gradient(
        total_heat_gain=P_T,
        conductor_thermal_conductivity=lambda_,
        core_diameter=D_c,
        conductor_diameter=D,
    )

    assert delta_T == approx(temperature_difference)


def test_thermal_conductivity_of_air_has_correct_roots():
    a = -2.763e-8
    b = 7.23e-5
    c = 2.368e-2

    tmp = np.sqrt(b**2 - 4 * a * c)
    root1 = (-b - tmp) / (2 * a)
    root2 = (-b + tmp) / (2 * a)

    assert convective_cooling.compute_thermal_conductivity_of_air(root1) == approx(0)
    assert convective_cooling.compute_thermal_conductivity_of_air(root2) == approx(0)


@hypothesis.given(air_temperature=st.floats(allow_nan=False, min_value=-273, max_value=1000))
def test_thermal_conductivity_of_air_has_correct_interpolant(air_temperature):
    T_f = np.arange(3) + air_temperature
    lambda_ = convective_cooling.compute_thermal_conductivity_of_air(T_f)
    lagrange_poly = lagrange(T_f, lambda_)

    a = -2.763e-8
    b = 7.23e-5
    c = 2.368e-2

    np.testing.assert_allclose(lagrange_poly.coef, [a, b, c])


@pytest.mark.parametrize(
    "air_temperature, air_thermal_conductivity",
    [(0, 2.368e-2), (10, 2.368e-2 + 7.23e-4 - 2.763e-6)],
)
def test_thermal_conductivity_of_air_with_examples(air_temperature, air_thermal_conductivity):
    T_f = air_temperature
    lambda_f = air_thermal_conductivity
    assert convective_cooling.compute_thermal_conductivity_of_air(T_f) == approx(lambda_f)


@hypothesis.given(air_temperature=st.floats(allow_nan=False, min_value=-273, max_value=1000))
def test_compute_air_density_scales_correctly_with_air_temperature(air_temperature):
    T_f = air_temperature
    y = 1

    numerator = 1.293 - 1.525e-4 + 6.379e-9
    assert convective_cooling.compute_air_density(T_f, y) == approx(numerator / (1 + 3.67e-3 * T_f))


@hypothesis.given(
    air_temperature=st.floats(allow_nan=False, min_value=-273, max_value=1000),
    height_above_sea_level=st.floats(allow_nan=False, min_value=0, max_value=10_000),
)
def test_compute_air_density_scales_has_correct_height_above_sea_level_interpolant(
    air_temperature, height_above_sea_level
):
    T_f = air_temperature
    y = np.arange(3) + height_above_sea_level
    gamma = convective_cooling.compute_air_density(T_f, y)
    lagrange_poly = lagrange(y, gamma)

    coefficients = np.array([6.379e-9, -1.525e-4, 1.293]) / (1 + 3.67e-3 * T_f)

    np.testing.assert_allclose(lagrange_poly.coef, coefficients)


@pytest.mark.parametrize(
    "height_above_sea_level, air_temperature, air_density",
    [
        (0, 0, 1.293),
        (1000, 0, 1.293 - 1.525e-1 + 6.379e-3),
        (1000, 0, 1.293 - 1.525e-1 + 6.379e-3),
        (1000, 10, (1.293 - 1.525e-1 + 6.379e-3) / 1.0367),
    ],
)
def test_compute_air_density_with_examples(height_above_sea_level, air_temperature, air_density):
    y = height_above_sea_level
    T_f = air_temperature
    gamma = air_density

    assert convective_cooling.compute_air_density(T_f, y) == pytest.approx(gamma)


@hypothesis.given(
    air_temperature=st.floats(allow_nan=False, min_value=-273, max_value=1000),
)
def test_dynamic_viscosity_of_air_has_correct_interpolant(air_temperature):
    T_f = np.arange(3) + air_temperature
    mu_f = convective_cooling.compute_dynamic_viscosity_of_air(T_f)

    lagrange_poly = lagrange(T_f, mu_f)
    coef = np.array([-2.03e-5, 4.635e-2, 17.239]) * 1e-6

    np.testing.assert_allclose(lagrange_poly.coef, coef)


def test_dynamic_viscosity_of_air_has_correct_roots():
    a, b, c = np.array([-2.03e-5, 4.635e-2, 17.239]) * 1e-6

    tmp = np.sqrt(b**2 - 4 * a * c)
    root1 = (-b - tmp) / (2 * a)
    root2 = (-b + tmp) / (2 * a)

    assert convective_cooling.compute_dynamic_viscosity_of_air(root1) == approx(0)
    assert convective_cooling.compute_dynamic_viscosity_of_air(root2) == approx(0)


@pytest.mark.parametrize(
    "air_temperature, dynamic_viscosity", [(0, 1.7239e-5), (0.5, 1.7239e-5 + 2.3175e-8 - 5.075e-12)]
)
def test_dynamic_viscosity_of_air_with_examples(air_temperature, dynamic_viscosity):
    T_f = air_temperature
    mu_f = dynamic_viscosity
    assert convective_cooling.compute_dynamic_viscosity_of_air(T_f) == approx(mu_f)


@hypothesis.given(dynamic_viscosity_of_air=st.floats(allow_nan=False, allow_infinity=False))
def test_kinematic_viscosity_of_air_scales_linearly_with_dynamic_viscosity(
    dynamic_viscosity_of_air,
):
    mu_f = dynamic_viscosity_of_air
    gamma = 1

    assert convective_cooling.compute_kinematic_viscosity_of_air(mu_f, gamma) == approx(mu_f)


@hypothesis.given(air_density=st.floats(min_value=1e-8, allow_nan=False, allow_infinity=False))
def test_kinematic_viscosity_of_air_scales_inversely_with_density(air_density):
    mu_f = 1
    gamma = air_density

    assert convective_cooling.compute_kinematic_viscosity_of_air(mu_f, gamma) == approx(1 / gamma)


def test_kinematic_viscosity_of_air_with_example():
    mu_f = 4
    gamma = 4

    assert convective_cooling.compute_kinematic_viscosity_of_air(mu_f, gamma) == approx(1)


# Tests for unitless quantities
###############################


@hypothesis.given(wind_speed=st.floats(min_value=0, max_value=1000, allow_nan=False))
def test_reynolds_number_scales_linearly_with_wind_speed(wind_speed):
    V = wind_speed
    D = 1
    Nu_f = 1

    assert convective_cooling.compute_reynolds_number(V, D, Nu_f) == pytest.approx(V)


@hypothesis.given(conductor_diameter=st.floats(min_value=0, max_value=1000, allow_nan=False))
def test_reynolds_number_scales_linearly_with_diameter(conductor_diameter):
    V = 1
    D = conductor_diameter
    Nu_f = 1

    assert convective_cooling.compute_reynolds_number(V, D, Nu_f) == pytest.approx(D)


@hypothesis.given(kinematic_viscosity=st.floats(min_value=1e-10, allow_nan=False))
def test_reynolds_number_scales_inversly_with_kinematic_viscosity(kinematic_viscosity):
    V = 1
    D = 1
    Nu_f = kinematic_viscosity

    assert convective_cooling.compute_reynolds_number(V, D, Nu_f) == approx(1 / Nu_f)


def test_reynolds_number_with_example():
    V = 0.1
    D = 1.2
    Nu_f = 10

    assert convective_cooling.compute_reynolds_number(V, D, Nu_f) == approx(0.012)


@hypothesis.given(conductor_diameter=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False))
def test_grashof_number_scales_cubic_with_conductor_diameter(conductor_diameter):
    D = conductor_diameter
    T_s = 1
    T_a = 0
    Nu_f = 1
    g = 273.65  # 0.5*T_s in Kelvin

    Gr = convective_cooling.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(D**3)


@hypothesis.given(surface_temperature=st.floats(min_value=273, max_value=1000, allow_nan=False))
def test_grashof_number_scales_correctly_with_surface_temperature(surface_temperature):
    D = 1
    T_s = surface_temperature
    T_a = 0
    Nu_f = 1
    g = 0.5 * T_s + 273.15

    Gr = convective_cooling.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(abs(T_s))


@hypothesis.given(air_temperature=st.floats(min_value=273, max_value=1000, allow_nan=False))
def test_grashof_number_scales_correctly_with_air_temperature(air_temperature):
    D = 1
    T_s = 0
    T_a = air_temperature
    Nu_f = 1
    g = 0.5 * T_a + 273.15

    Gr = convective_cooling.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(abs(T_a))


@hypothesis.given(
    kinematic_viscosity_of_air=st.floats(min_value=1e-5, max_value=1e10, allow_nan=False)
)
def test_grashof_number_scales_inversely_squared_with_kinematic_viscosity(
    kinematic_viscosity_of_air,
):
    D = 1
    T_s = 1
    T_a = 0
    Nu_f = kinematic_viscosity_of_air
    g = 273.65  # 0.5*T_s in Kelvin

    Gr = convective_cooling.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(1 / (Nu_f**2))


@hypothesis.given(coefficient_of_gravity=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False))
def test_grashof_number_scales_linearly_with_gravity(coefficient_of_gravity):
    D = 1
    T_s = 1
    T_a = 0
    Nu_f = np.sqrt(1 / 273.65)
    g = coefficient_of_gravity

    Gr = convective_cooling.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(g)


def test_grashof_number_with_example():
    D = 4
    T_s = 4
    T_a = 2
    Nu_f = 2
    g = 276.15

    Gr = convective_cooling.compute_grashof_number(D, T_s, T_a, Nu_f, g)
    assert Gr == approx(32)


@hypothesis.given(thermal_conductivity=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False))
def test_prandtl_number_scales_inversely_with_thermal_conductivity(thermal_conductivity):
    c_f = 1
    mu_f = 1
    lambda_f = thermal_conductivity

    assert convective_cooling.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(1 / lambda_f)


@hypothesis.given(dynamic_viscosity=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False))
def test_prandtl_number_scales_linearly_with_dynamic_viscosity(dynamic_viscosity):
    c_f = 1
    mu_f = dynamic_viscosity
    lambda_f = 1

    assert convective_cooling.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(mu_f)


@hypothesis.given(
    specific_heat_capacity=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False)
)
def test_prandtl_number_scales_linearly_with_specific_heat_capacity(specific_heat_capacity):
    c_f = specific_heat_capacity
    mu_f = 1
    lambda_f = 1

    assert convective_cooling.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(c_f)


def test_compute_prandtl_number_with_example():
    c_f = 0.5
    mu_f = 3
    lambda_f = 0.5

    assert convective_cooling.compute_prandtl_number(lambda_f, mu_f, c_f) == approx(3)


## Nusselt number calculation
#############################

_eps = np.finfo(np.float64).eps


@pytest.mark.parametrize(
    "reynolds_number, conductor_roughness, B, n,",
    # Smooth conductors
    [(0, Rs, 0, 0) for Rs in [0, np.nan]]
    + [(34, Rs, 0, 0) for Rs in [0, np.nan]]
    + [(35, Rs, 0.583, 0.471) for Rs in [0, np.nan]]
    + [(2_500, Rs, 0.583, 0.471) for Rs in [0, np.nan]]
    + [(4_999, Rs, 0.583, 0.471) for Rs in [0, np.nan]]
    + [(5_000, Rs, 0.148, 0.633) for Rs in [0, np.nan]]
    + [(25_000, Rs, 0.148, 0.633) for Rs in [0, np.nan]]
    + [(49_999, Rs, 0.148, 0.633) for Rs in [0, np.nan]]
    + [(50_000, Rs, 0.0208, 0.814) for Rs in [0, np.nan]]
    + [(100_000, Rs, 0.0208, 0.814) for Rs in [0, np.nan]]
    + [(199_999, Rs, 0.0208, 0.814) for Rs in [0, np.nan]]
    # Stranded smooth conductors
    + [(0, Rs, 0, 0) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    + [(99, Rs, 0, 0) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    + [(100, Rs, 0.641, 0.471) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    + [(2_500, Rs, 0.641, 0.471) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    + [(2_649, Rs, 0.641, 0.471) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    + [(2_650, Rs, 0.178, 0.633) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    + [(25_000, Rs, 0.178, 0.633) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    + [(49_999, Rs, 0.178, 0.633) for Rs in np.linspace(_eps, 0.05, 5, endpoint=False)]
    # Stranded rough conductors
    + [(0, Rs, 0, 0) for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)]
    + [(99, Rs, 0, 0) for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)]
    + [
        (100, Rs, 0.641, 0.471)
        for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)
    ]
    + [
        (2_500, Rs, 0.641, 0.471)
        for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)
    ]
    + [
        (2_649, Rs, 0.641, 0.471)
        for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)
    ]
    + [
        (2_650, Rs, 0.048, 0.800)
        for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)
    ]
    + [
        (25_000, Rs, 0.048, 0.800)
        for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)
    ]
    + [
        (49_999, Rs, 0.048, 0.800)
        for Rs in np.linspace(np.nextafter(0.05, 1), 1000, 5, endpoint=False)
    ],
)
def test_perpendicular_flow_nusselt_number_uses_correct_exponential(
    reynolds_number,
    conductor_roughness,
    B,
    n,
):
    Re = reynolds_number
    Rs = conductor_roughness
    h = 0.5

    if B == 0:
        assert convective_cooling.compute_perpendicular_flow_nusseltnumber(Re, Rs) == approx(0)
        return

    Nu_90_0 = convective_cooling.compute_perpendicular_flow_nusseltnumber(Re, Rs)
    Nu_90_1 = convective_cooling.compute_perpendicular_flow_nusseltnumber(Re + h, Rs)

    n_est = (np.log(Nu_90_1) - np.log(Nu_90_0)) / (np.log(Re + h) - np.log(Re))
    B_est = Nu_90_0 / (Re**n)
    assert n_est == approx(n, rel=1e-8)
    assert B_est == approx(B, rel=1e-8)


@hypothesis.given(
    perpendicular_flow_nusselt_number=st.floats(
        min_value=1e-10, allow_infinity=False, allow_nan=False
    ),
    angle_of_attack=st.floats(allow_infinity=False, min_value=0, max_value=90),
)
def test_stranded_angle_of_attack_correction_has_correct_form(
    perpendicular_flow_nusselt_number,
    angle_of_attack,
):
    Nu_90 = perpendicular_flow_nusselt_number
    delta = np.radians(angle_of_attack)
    Nu_delta = convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, delta, conductor_roughness=1
    )

    if angle_of_attack <= 24:
        assert Nu_delta / Nu_90 == approx(0.42 + 0.68 * np.sin(delta) ** 1.08, rel=1e-8)
    else:
        assert Nu_delta / Nu_90 == approx(0.42 + 0.58 * np.sin(delta) ** 0.90, rel=1e-8)


@hypothesis.given(
    perpendicular_flow_nusselt_number=st.floats(
        min_value=1e-10, allow_infinity=False, allow_nan=False
    ),
    angle_of_attack=st.floats(allow_infinity=False, min_value=0, max_value=90),
)
@pytest.mark.parametrize("conductor_roughness", [0, np.nan])
def test_smooth_angle_of_attack_correction_has_correct_form(
    perpendicular_flow_nusselt_number,
    angle_of_attack,
    conductor_roughness,
):
    Nu_90 = perpendicular_flow_nusselt_number
    delta = np.radians(angle_of_attack)
    Rs = conductor_roughness

    Nu_delta = convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, delta, conductor_roughness=Rs
    )

    assert Nu_delta / Nu_90 == approx(
        (np.sin(delta) ** 2 + 0.0169 * np.cos(delta) ** 2) ** 0.225, rel=1e-8
    )


@pytest.mark.parametrize(
    "perpendicular_flow_nusselt_number, angle_of_attack, conductor_roughness, corrected_nusselt_number",  # noqa
    [
        (0, np.pi / 6, 0, 0),
        (0, np.pi / 6, np.nan, 0),
        (0, np.pi / 6, _eps, 0),
        (0, np.pi / 6, 1, 0),
        (0, np.arcsin(0.1), _eps, 0),
        (0, np.arcsin(0.1), 1, 0),
        (0.5, np.pi / 6, 0, 0.5 * (0.0169 + 0.25 * (1 - 0.0169)) ** 0.225),
        (0.5, np.pi / 6, np.nan, 0.5 * (0.0169 + 0.25 * (1 - 0.0169)) ** 0.225),
        (0.5, np.pi / 6, _eps, 0.21 + 0.29 * (0.5**0.9)),
        (0.5, np.pi / 6, 1, 0.21 + 0.29 * (0.5**0.9)),
        (0.5, np.arcsin(0.1), _eps, 0.21 + 0.34 * (0.1**1.08)),
        (0.5, np.arcsin(0.1), 1, 0.21 + 0.34 * (0.1**1.08)),
    ],
)
def test_angle_of_attack_correction_with_examples(
    perpendicular_flow_nusselt_number,
    angle_of_attack,
    conductor_roughness,
    corrected_nusselt_number,
):
    Nu_90 = perpendicular_flow_nusselt_number
    delta = angle_of_attack
    Nu_delta = corrected_nusselt_number
    Rs = conductor_roughness

    Nu_delta_est = convective_cooling.correct_wind_direction_effect_on_nusselt_number(
        Nu_90, delta, Rs
    )
    assert Nu_delta_est == approx(Nu_delta, rel=1e-8)


@pytest.mark.parametrize(
    "x, A, m",
    (
        [(x, 1.020, 0.148) for x in np.logspace(-1, 2, 5, endpoint=False)]
        + [(x, 0.850, 0.188) for x in np.logspace(2, 4, 5, endpoint=False)]
        + [(x, 0.480, 0.250) for x in np.logspace(4, 7, 5, endpoint=False)]
        + [(x, 0.125, 0.333) for x in np.logspace(7, 12, 5, endpoint=True)]
    ),
)
def test_horizontal_natural_nusselt_number_uses_correct_exponential(x, A, m):
    Nu_0 = A * (x**m)

    assert convective_cooling.compute_horizontal_natural_nusselt_number(x, 1) == approx(
        Nu_0, rel=1e-8
    )
    assert convective_cooling.compute_horizontal_natural_nusselt_number(1, x) == approx(
        Nu_0, rel=1e-8
    )


@hypothesis.given(inclination=st.floats(min_value=0, max_value=np.pi / 3))
@pytest.mark.parametrize("conductor_roughness", [0, np.nan])
def test_smooth_inclination_correction_has_correct_form(inclination, conductor_roughness):
    Nu_0 = 1
    beta = inclination
    Rs = conductor_roughness

    Nu_beta = convective_cooling.correct_natural_nusselt_number_inclination(Nu_0, inclination, Rs)
    assert 1 - Nu_beta == approx(1.58e-4 * np.degrees(beta) ** 1.5, rel=1e-8)


@hypothesis.given(inclination=st.floats(min_value=0, max_value=np.radians(80)))
@pytest.mark.parametrize("conductor_roughness", [_eps, 1])
def test_stranded_inclination_correction_has_correct_form(inclination, conductor_roughness):
    Nu_0 = 1
    beta = inclination
    Rs = conductor_roughness

    Nu_beta = convective_cooling.correct_natural_nusselt_number_inclination(Nu_0, inclination, Rs)
    assert 1 - Nu_beta == approx(1.76e-6 * np.degrees(beta) ** 2.5, rel=1e-8)


@pytest.mark.parametrize(
    "horizontal_natural_nusselt_number, inclination, conductor_roughness, natural_nusselt_number",
    (
        [0.5, np.radians(1), 0, 0.5 * (1 - 1.58e-4)],
        [0.5, np.radians(1), np.nan, 0.5 * (1 - 1.58e-4)],
        [0.5, np.radians(1), _eps, 0.5 * (1 - 1.76e-6)],
        [0.5, np.radians(1), 1, 0.5 * (1 - 1.76e-6)],
    ),
)
def test_inclination_correction_with_examples(
    horizontal_natural_nusselt_number, inclination, conductor_roughness, natural_nusselt_number
):
    Nu_0 = horizontal_natural_nusselt_number
    beta = inclination
    Nu_nat = natural_nusselt_number
    Rs = conductor_roughness
    Nu_nat_est = convective_cooling.correct_natural_nusselt_number_inclination(Nu_0, beta, Rs)

    assert Nu_nat_est == approx(Nu_nat, rel=1e-8)


@hypothesis.given(
    forced_convection_nusselt_number=st.floats(allow_nan=False),
    natural_nusselt_number=st.floats(allow_nan=False),
)
def test_compute_nusselt_number(forced_convection_nusselt_number, natural_nusselt_number):
    Nu_delta = forced_convection_nusselt_number
    Nu_beta = natural_nusselt_number

    Nu = max(Nu_delta, Nu_beta)
    Nu_est = convective_cooling.compute_nusselt_number(Nu_delta, Nu_beta)

    assert Nu == Nu_est


@hypothesis.given(
    nusselt_number=st.floats(
        min_value=0,
        max_value=1e10,
        allow_nan=False,
    )
)
def test_convective_cooling_scales_linearly_with_nusselt_number(nusselt_number):
    Nu = nusselt_number
    T_s = 1
    T_a = 0
    lambda_f = 1 / np.pi

    assert convective_cooling.compute_convective_cooling(T_s, T_a, Nu, lambda_f) == approx(Nu)


@hypothesis.given(
    thermal_conductivity_of_air=st.floats(
        min_value=0,
        max_value=1e10,
        allow_nan=False,
    )
)
def test_convective_cooling_scales_linearly_with_thermal_conductivity_of_air(
    thermal_conductivity_of_air,
):
    Nu = 1 / np.pi
    T_s = 1
    T_a = 0
    lambda_f = thermal_conductivity_of_air

    assert convective_cooling.compute_convective_cooling(T_s, T_a, Nu, lambda_f) == approx(lambda_f)


@hypothesis.given(
    surface_temperature=st.floats(
        min_value=-1e10,
        max_value=1e10,
        allow_nan=False,
    ),
    air_temperature=st.floats(
        min_value=-1e10,
        max_value=1e10,
        allow_nan=False,
    ),
)
def test_convective_cooling_scales_linearly_with_temperature_difference(
    surface_temperature, air_temperature
):
    Nu = 1
    T_s = surface_temperature
    T_a = air_temperature
    lambda_f = 1 / np.pi

    assert convective_cooling.compute_convective_cooling(T_s, T_a, Nu, lambda_f) == approx(
        T_s - T_a
    )


@hypothesis.given(
    air_temperature=st.floats(
        min_value=-1e10,
        max_value=1e10,
        allow_nan=False,
    )
)
def test_convective_cooling_scales_affinely_with_air_temperature(air_temperature):
    Nu = 1
    T_s = 1
    T_a = air_temperature
    lambda_f = 1 / np.pi

    assert 1 - convective_cooling.compute_convective_cooling(T_s, T_a, Nu, lambda_f) == approx(T_a)


@hypothesis.given(
    surface_temperature=st.floats(
        min_value=-1e10,
        max_value=1e10,
        allow_nan=False,
    ),
)
def test_convective_cooling_scales_affinely_with_surface_temperature(surface_temperature):
    Nu = 1
    T_s = surface_temperature
    T_a = 1
    lambda_f = 1 / np.pi

    assert convective_cooling.compute_convective_cooling(T_s, T_a, Nu, lambda_f) + 1 == approx(T_s)
