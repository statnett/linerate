import hypothesis
import hypothesis.strategies as st
import numpy as np
from pytest import approx
from scipy.interpolate import lagrange

import linerate.equations.joule_heating as joule_heating


@hypothesis.given(
    conductor_temperature=st.floats(min_value=-273, max_value=400, allow_nan=False),
    slope=st.floats(min_value=0, max_value=1e3, allow_nan=False),
    intercept=st.floats(min_value=0, max_value=1e3, allow_nan=False),
    temperatures=st.tuples(
        st.floats(min_value=-273, max_value=300, allow_nan=False),
        st.floats(min_value=-273, max_value=300, allow_nan=False),
    ).filter(lambda tup: abs(tup[0] - tup[1]) > 0.1),
)
def test_resistance_has_correct_interpolant(
    conductor_temperature,
    slope,
    intercept,
    temperatures,
):
    T_c = 5 * np.arange(2) + conductor_temperature
    T1, T2 = temperatures
    a = slope
    b = intercept
    R1 = a * T1 + b
    R2 = a * T2 + b

    R = joule_heating.compute_resistance(T_c, T1, T2, R1, R2)
    lagrange_poly = lagrange(T_c, R)

    if len(lagrange_poly.coef) == 1:
        assert lagrange_poly.coef[0] == approx(b)
        assert a == approx(0)
    else:
        np.testing.assert_allclose(lagrange_poly.coef, [a, b], rtol=1e-5, atol=1e-8)


@hypothesis.given(current=st.floats(min_value=0, max_value=1e10, allow_nan=False))
def test_acsr_magnetic_core_loss_correction_is_affine_in_current_without_saturation(current):
    R_ac = 1
    I = current  # noqa
    A = 1
    b = 1
    m = 1
    max_relative_increase = float("inf")

    R_est = joule_heating.correct_resistance_acsr_magnetic_core_loss(
        R_ac, I, A, b, m, max_relative_increase
    )
    assert R_est / R_ac == approx(1 + I)


@hypothesis.given(resistance=st.floats(min_value=0, max_value=1e10, allow_nan=False))
def test_acsr_magnetic_core_loss_correction_is_linear_in_resistance(resistance):
    R_ac = resistance
    I = 1  # noqa
    A = 1
    b = 1
    m = 1
    max_relative_increase = float("inf")

    R_est = joule_heating.correct_resistance_acsr_magnetic_core_loss(
        R_ac, I, A, b, m, max_relative_increase
    )
    assert R_est == approx(2 * R_ac)


@hypothesis.given(constant_magnetic_effect=st.floats(min_value=0, max_value=1e10, allow_nan=False))
def test_acsr_magnetic_core_loss_correction_is_affine_in_constant_magnetic_effect(
    constant_magnetic_effect,
):
    R_ac = 2
    I = 1  # noqa
    A = 1
    b = constant_magnetic_effect
    m = 1
    max_relative_increase = float("inf")

    R_est = joule_heating.correct_resistance_acsr_magnetic_core_loss(
        R_ac, I, A, b, m, max_relative_increase
    )
    assert R_est / R_ac == approx(b + 1)


@hypothesis.given(
    current_density_proportional_magnetic_effect=st.floats(
        min_value=0, max_value=1e10, allow_nan=False
    )
)
def test_acsr_magnetic_core_loss_correction_is_affine_in_linear_magnetic_effect(
    current_density_proportional_magnetic_effect,
):
    R_ac = 2
    I = 1  # noqa
    A = 1
    m = current_density_proportional_magnetic_effect
    b = 1
    max_relative_increase = float("inf")

    R_est = joule_heating.correct_resistance_acsr_magnetic_core_loss(
        R_ac, I, A, b, m, max_relative_increase
    )
    assert R_est / R_ac == approx(1 + m)


@hypothesis.given(max_relative_increase=st.floats(min_value=0, max_value=1e10, allow_nan=False))
def test_acsr_magnetic_core_loss_correction_saturates(max_relative_increase):
    R_ac = 2
    I = 2e10  # noqa
    A = 1
    m = 1
    b = 1

    R_est = joule_heating.correct_resistance_acsr_magnetic_core_loss(
        R_ac, I, A, m, b, max_relative_increase
    )
    assert R_est / R_ac == approx(max_relative_increase)


@hypothesis.given(
    current=st.floats(min_value=0, max_value=1e5, allow_nan=False),
    resistance=st.floats(min_value=0, max_value=1e5, allow_nan=False),
)
def test_joule_heating(current, resistance):
    I = current  # noqa
    R = resistance

    assert joule_heating.compute_joule_heating(I, R) == approx(R * (I**2))
