import hypothesis
import hypothesis.strategies as st
import numpy as np

from pytest import approx
from linerate.equations import convective_cooling


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
