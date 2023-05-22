"""
This submodule contains implementations of mathematical equations. Each function is implemented as
a pure function corresponding to one or more equations. The left hand side of the equations are
returned and the parameters of the right hand side are required as function arguments. Each
function is implemented with vectorization over all numerical parameters.
"""

from . import cigre601, ieee738, joule_heating, math, radiative_cooling, solar_angles  # noqa
