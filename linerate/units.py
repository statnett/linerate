from typing import Annotated, Union

import numpy as np
import numpy.typing as npt

FloatOrFloatArray = Union[float, np.float64, npt.NDArray[np.float64]]

OhmPerMeter = Annotated[FloatOrFloatArray, "Ω/m"]
PerCelsius = Annotated[FloatOrFloatArray, "1/°C"]
PerSquareCelsius = Annotated[FloatOrFloatArray, "1/°C²"]
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
SquareMeterPerSecond = Annotated[FloatOrFloatArray, "m²/s"]
KilogramPerMeterPerSecond = Annotated[FloatOrFloatArray, "kg/(m s)"]
KilogramPerCubeMeter = Annotated[FloatOrFloatArray, "kg/m³"]

Date = Union[np.datetime64, npt.NDArray[np.datetime64]]
