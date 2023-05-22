from typing import Union

try:
    from typing import Annotated
except ImportError:  # Python version <3.9
    from typing_extensions import Annotated

import numpy as np
import numpy.typing as npt

FloatOrFloatArray = Union[float, np.float64, npt.NDArray[np.float64]]
BoolOrBoolArray = Union[bool, np.bool_, npt.NDArray[np.bool_]]

OhmPerMeter = Annotated[FloatOrFloatArray, "Ω/m"]
Ampere = Annotated[FloatOrFloatArray, "A"]
Radian = Annotated[FloatOrFloatArray, "rad"]
Degrees = Annotated[FloatOrFloatArray, "°"]
Kelvin = Annotated[FloatOrFloatArray, "K"]
Celsius = Annotated[FloatOrFloatArray, "°C"]
Meter = Annotated[FloatOrFloatArray, "m"]
MeterPerSecond = Annotated[FloatOrFloatArray, "m/s"]
MeterPerSquareSecond = Annotated[FloatOrFloatArray, "m/s²"]
SquareMeterPerAmpere = Annotated[FloatOrFloatArray, "m²/A"]
SquareMeter = Annotated[FloatOrFloatArray, "m²"]
Unitless = Annotated[FloatOrFloatArray, ""]
JoulePerKilogramPerKelvin = Annotated[FloatOrFloatArray, "J/(kg K)"]
WattPerSquareMeter = Annotated[FloatOrFloatArray, "W/m²"]
WattPerMeter = Annotated[FloatOrFloatArray, "W/m"]
WattPerMeterPerKelvin = Annotated[FloatOrFloatArray, "W/(m K)"]
WattPerMeterPerCelsius = Annotated[FloatOrFloatArray, "W/(m °C)"]
SquareMeterPerSecond = Annotated[FloatOrFloatArray, "m²/s"]
KilogramPerMeterPerSecond = Annotated[FloatOrFloatArray, "kg/(m s)"]
KilogramPerCubeMeter = Annotated[FloatOrFloatArray, "kg/m³"]

Date = Union[np.datetime64, npt.NDArray[np.datetime64]]
