from pytest import approx

import hypothesis
import hypothesis.strategies as st

from linerate.units import Celsius, KilogramPerMeter
from linerate.equations import heat_capacity


@hypothesis.given(mass_per_unit_length=st.floats(min_value=0, max_value=1e10, allow_nan=False))
def test_heat_capacity_is_linear_in_mass_per_unit_length(mass_per_unit_length: KilogramPerMeter):
    m = mass_per_unit_length
    c = 1
    beta = 1
    T = 20
    mc = heat_capacity.calculate_heat_capacity_per_unit_length(m, c, beta, T)
    assert mc == approx(m)


@hypothesis.given(
    specific_heat_capacity_at_20_celsius=st.floats(min_value=0, max_value=1e10, allow_nan=False)
)
def test_heat_capacity_is_linear_in_specific_heat_capacity(
    specific_heat_capacity_at_20_celsius: KilogramPerMeter,
):
    m = 1
    c = specific_heat_capacity_at_20_celsius
    beta = 1
    T = 20
    mc = heat_capacity.calculate_heat_capacity_per_unit_length(m, c, beta, T)
    assert mc == approx(c)


@hypothesis.given(
    specific_heat_capacity_coefficient=st.floats(min_value=0, max_value=1e10, allow_nan=False)
)
def test_heat_capacity_is_linear_in_heat_capacity_coefficient(
    specific_heat_capacity_coefficient: KilogramPerMeter,
):
    m = 1
    c = 1
    beta = specific_heat_capacity_coefficient
    T = 21
    mc = heat_capacity.calculate_heat_capacity_per_unit_length(m, c, beta, T)
    assert mc == approx(beta + 1)


@hypothesis.given(temperature=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False))
def test_heat_capacity_is_linear_in_temperature(temperature: Celsius):
    m = 1
    c = 1
    beta = 1
    T = temperature
    mc = heat_capacity.calculate_heat_capacity_per_unit_length(m, c, beta, T)
    assert mc == approx(T - 19)
