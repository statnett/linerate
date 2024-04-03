import pytest

import linerate.solver as solver


def test_compute_conductor_temperature_computes_correct_temperature():
    def heat_balance(conductor_temperature, current):
        A = current
        T = conductor_temperature
        return (A - 100 * T) * (current + 100 * T)

    conductor_temperature = solver.compute_conductor_temperature(
        heat_balance, current=1500, min_temperature=0, max_temperature=150, tolerance=1e-8
    )
    assert conductor_temperature == pytest.approx(15, rel=1e-7)


def test_compute_conductor_temperature_computes_correct_ampacity():
    def heat_balance(conductor_temperature, current):
        A = current
        T = conductor_temperature
        return (A - 100 * T) * (current + 100 * T)

    conductor_temperature = solver.compute_conductor_ampacity(
        heat_balance,
        max_conductor_temperature=90,
        min_ampacity=0,
        max_ampacity=10_000,
        tolerance=1e-8,
    )
    assert conductor_temperature == pytest.approx(9000, rel=1e-7)


def test_bisect_raises_value_error():
    def heat_balance(current):
        A = current
        T = 90
        return (A + 100 * T) * (A + 100 * T)

    with pytest.raises(ValueError):
        solver.bisect(
            heat_balance,
            xmin=0,
            xmax=10_000,
            tolerance=1e-8,
        )
