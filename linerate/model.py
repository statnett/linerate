"""
These thermal model classes provide an easy-to-use interface to compute the thermal rating.
They store conductor and span metadata as well as the weather parameters, compute all the
heating and cooling effects and use those to estimate the thermal rating and conductor
temperature. All numerical heavy-lifting is handled by the ``linerate.equations`` and the
``linerate.solver`` modules.
"""

from linerate.models.cigre207 import Cigre207
from linerate.models.cigre601 import Cigre601
from linerate.models.ieee738 import IEEE738
from linerate.models.thermal_model import ThermalModel

__all__ = ["ThermalModel", "Cigre601", "IEEE738", "Cigre207"]
