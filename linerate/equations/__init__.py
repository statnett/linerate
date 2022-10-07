"""
This submodule contains implementations of equations listed in :cite:p:`cigre601`. Each function is
implemented as a pure function corresponding to one or more equations. The left hand side is always
returned and the parameters of the right hand side is required as function arguments. Each function
is implemented with vectorization over all numerical parameters (except the diameter parameters).
"""

from . import convective_cooling, joule_heating, math, radiative_cooling, solar_heating  # noqa
