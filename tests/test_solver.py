from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

import linerate.solver as solver
from linerate.units import Ampere, Celsius, Duration, WattPerMeter


def test_compute_conductor_temperature_computes_correct_temperature(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
):
    conductor_temperature = solver.compute_conductor_temperature(
        heat_balance, current=1500, min_temperature=0, max_temperature=150, tolerance=1e-8
    )
    assert conductor_temperature == pytest.approx(15, rel=1e-7)


def test_compute_conductor_ampacity_computes_correct_ampacity(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
):
    conductor_temperature = solver.compute_conductor_ampacity(
        heat_balance,
        max_conductor_temperature=90,
        min_ampacity=0,
        max_ampacity=10_000,
        tolerance=1e-8,
    )
    assert conductor_temperature == pytest.approx(9000, rel=1e-7)


def test_compute_conductor_ampacity_returns_zero_for_elements_with_positive_heat_balance_at_zero_current():
    def heat_balance(conductor_temperature: Celsius, current: Ampere) -> WattPerMeter:
        return current**2 - np.array([100, -1, 100])

    ampacity = solver.compute_conductor_ampacity(
        heat_balance, max_conductor_temperature=90, tolerance=1e-8
    )

    np.testing.assert_array_almost_equal(ampacity, [10, 0, 10], decimal=8)


def test_compute_conductor_transient_ampacity_returns_zero_for_elements_with_positive_heat_balance_at_zero_current():
    def final_temperature(
        initial_conductor_temperature: Celsius, heating_duration: Duration, current: Ampere
    ) -> Celsius:
        return 90 + current**2 - np.array([100, -1, 100])

    ampacity = solver.compute_conductor_transient_ampacity(
        final_temperature,
        max_conductor_temperature=90,
        initial_conductor_temperature=20,
        heating_duration=np.timedelta64(600, "s"),
        tolerance=1e-8,
    )

    np.testing.assert_array_almost_equal(ampacity, [10, 0, 10], decimal=8)


def test_compute_conductor_ampacity_raises_when_solution_exceeds_max_ampacity():
    def heat_balance(conductor_temperature: Celsius, current: Ampere) -> WattPerMeter:
        return current**2 - 100

    with pytest.raises(ValueError):
        solver.compute_conductor_ampacity(
            heat_balance, max_conductor_temperature=90, max_ampacity=5
        )


def test_compute_conductor_temperature_raises_when_solution_is_below_min_ampacity():
    def heat_balance(conductor_temperature: Celsius, current: Ampere) -> WattPerMeter:
        return current**2 - 100

    with pytest.raises(ValueError):
        solver.compute_conductor_ampacity(
            heat_balance, max_conductor_temperature=90, min_ampacity=15
        )


def test_bisect_raises_value_error():
    def heat_balance(current):
        I = current  # noqa: E741
        T = 90
        return (I + 100 * T) * (I + 100 * T)

    with pytest.raises(ValueError):
        solver.bisect(
            heat_balance,
            xmin=0,
            xmax=10_000,
            tolerance=1e-8,
        )


def test_bisect_handles_function_returning_array_happy_path(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
):
    _heat_balance = partial(heat_balance, 90)

    solution = solver.bisect(
        _heat_balance,
        xmin=np.array([0, 0]),
        xmax=np.array([10_000, 10_000]),
        tolerance=1e-8,
    )
    np.testing.assert_array_almost_equal(solution, [9_000, 9_000], decimal=8)


def test_bisect_raises_valueerror_when_same_sign_for_array_input(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
):
    _heat_balance = partial(heat_balance, 90)
    with pytest.raises(ValueError):
        solver.bisect(
            _heat_balance,
            xmin=np.array([0, 0]),
            xmax=np.array([10_000, 8_000]),
            tolerance=1e-8,
        )


def test_bisect_raises_valueerror_when_infinite_in_array_input():
    with pytest.raises(ValueError):
        solver.bisect(
            lambda x: x,
            xmin=np.array([-np.inf, 0]),
            xmax=np.array([10_000, 10_000]),
            tolerance=1e-8,
        )


def test_bisect_returns_dtype_float_if_not_accept_invalid_values(
    heat_balance: Callable[[Celsius, Ampere], WattPerMeter],
):
    _heat_balance = partial(heat_balance, 90)

    solution = solver.bisect(
        _heat_balance,
        xmin=np.array([0, 0]),
        xmax=np.array([10_000, 10_000]),
        tolerance=1e-8,
        accept_invalid_values=False,
    )

    assert isinstance(solution, np.ndarray)
    assert solution.dtype == np.float64


def test_bisect_return_nan_if_heat_balance_returns_nan():
    def heat_balance(current: Ampere) -> WattPerMeter:
        return np.ones_like(current) * np.nan

    solution = solver.bisect(
        heat_balance,
        xmin=np.array([0, 0]),
        xmax=np.array([10_000, 10_000]),
        tolerance=1e-8,
    )

    np.testing.assert_array_equal(np.isnan(solution), np.full_like(solution, True, dtype=bool))
