import importlib.metadata as importlib_metadata

__version__ = importlib_metadata.version(__name__)

from . import equations  # noqa
from .model import *  # noqa
from .solver import *  # noqa
from .types import *  # noqa
