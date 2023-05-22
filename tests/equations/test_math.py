import hypothesis
import hypothesis.strategies as st
import numpy as np
from pytest import approx

import linerate.equations.math as cigre_math


@hypothesis.given(angle=st.floats())
def test_switch_cos_sin(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    sin_from_cos = cigre_math.switch_cos_sin(cos)
    cos_from_sin = cigre_math.switch_cos_sin(sin)

    if np.isnan(angle) or np.isinf(angle):
        assert np.isnan(sin_from_cos)
        assert np.isnan(cos_from_sin)
    else:
        assert np.abs(cos) == approx(cos_from_sin, abs=1e-8)
        assert np.abs(sin) == approx(sin_from_cos, abs=1e-8)
