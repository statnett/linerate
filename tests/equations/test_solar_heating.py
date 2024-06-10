import hypothesis
import hypothesis.strategies as st
from pytest import approx

from linerate.equations import solar_heating


def test_matches_solar_heating_example_from_cigre207():
    # See Appendix 1, Example 1 in Cigre 207
    D = 0.0286
    global_solar_radiation = 980
    absorptivity = 0.5
    heating = solar_heating.compute_solar_heating(absorptivity, global_solar_radiation, D)
    assert heating == approx(14.02, rel=1e-3)


@hypothesis.given(conductor_diameter=st.floats(allow_nan=False))
def test_solar_heating_scales_linearly_with_conductor_diameter(conductor_diameter):
    D = conductor_diameter
    alpha_s = 1
    I_T = 1

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(D)


@hypothesis.given(solar_absorptivity=st.floats(allow_nan=False))
def test_solar_heating_scales_linearly_with_solar_absorptivity(solar_absorptivity):
    D = 1
    alpha_s = solar_absorptivity
    I_T = 1

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(alpha_s)


@hypothesis.given(global_radiation_intensity=st.floats(allow_nan=False))
def test_solar_heating_scales_linearly_with_global_radiation_intensity(global_radiation_intensity):
    D = 1
    alpha_s = 1
    I_T = global_radiation_intensity

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(
        global_radiation_intensity
    )


def test_solar_heating_scales_linearly_with_example():
    D = 0.6
    alpha_s = 0.5
    I_T = 2

    assert solar_heating.compute_solar_heating(alpha_s, I_T, D) == approx(0.6)
