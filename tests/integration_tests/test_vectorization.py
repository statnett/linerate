import hypothesis
from hypothesis.extra.numpy import mutually_broadcastable_shapes, BroadcastableShapes
import numpy as np
import numpy.typing as npt

from linerate.models.cigre601 import Cigre601
from linerate.types import Conductor, ConductorWithHeatCapacity, Span, Tower, Weather

NUMBER_OF_VARIABLES_TRANSIENT = 32
NUMBER_OF_VARIABLES_STEADY_STATE = 24


class Shapes:
    def __init__(self, shapes: tuple[tuple[int, ...], ...]):
        self.shapes = shapes
        self.index = 0

    def get_zeros(self, dtype: np.dtype = np.dtype("f8")) -> npt.NDArray:
        out = np.zeros(self.shapes[self.index], dtype=dtype)
        self.index += 1
        return out


@hypothesis.given(
    array_shapes=mutually_broadcastable_shapes(
        num_shapes=NUMBER_OF_VARIABLES_STEADY_STATE, max_dims=5
    )
)
def test_vectorization(array_shapes: BroadcastableShapes):
    shapes = Shapes(array_shapes.input_shapes)

    conductor = Conductor(
        core_diameter=1,
        conductor_diameter=shapes.get_zeros() + 2,
        outer_layer_strand_diameter=shapes.get_zeros(),
        emissivity=shapes.get_zeros(),
        solar_absorptivity=shapes.get_zeros(),
        temperature1=shapes.get_zeros(),
        temperature2=shapes.get_zeros() + 1,
        resistance_at_temperature1=shapes.get_zeros(),
        resistance_at_temperature2=shapes.get_zeros(),
        aluminium_cross_section_area=1 + shapes.get_zeros(),
        constant_magnetic_effect=shapes.get_zeros(),
        current_density_proportional_magnetic_effect=shapes.get_zeros() + 1,
        max_magnetic_core_relative_resistance_increase=shapes.get_zeros(),
    )
    weather = Weather(
        air_temperature=shapes.get_zeros(),
        wind_direction=shapes.get_zeros(),
        wind_speed=shapes.get_zeros(),
        ground_albedo=shapes.get_zeros(),
        clearness_ratio=shapes.get_zeros(),
    )
    span = Span(
        conductor=conductor,
        start_tower=Tower(
            shapes.get_zeros(),
            shapes.get_zeros(),
            shapes.get_zeros(),
        ),
        end_tower=Tower(
            0.1 + shapes.get_zeros(),
            0.1 + shapes.get_zeros(),
            10 + shapes.get_zeros(),
        ),
        num_conductors=1,
    )
    time = shapes.get_zeros(dtype=np.dtype("datetime64[D]"))

    model = Cigre601(span=span, weather=weather, time=time)
    temperature = model.compute_conductor_temperature(100)
    assert isinstance(temperature, np.ndarray)
    assert temperature.shape == array_shapes.result_shape


@hypothesis.given(
    array_shapes=mutually_broadcastable_shapes(num_shapes=NUMBER_OF_VARIABLES_TRANSIENT, max_dims=5)
)
def test_transient_vectorization(array_shapes: BroadcastableShapes):
    shapes = Shapes(array_shapes.input_shapes)

    conductor = ConductorWithHeatCapacity(
        core_diameter=1,  # Core diameter is not used in any calculations so its shape will not be broadcast
        conductor_diameter=shapes.get_zeros() + 2,
        outer_layer_strand_diameter=shapes.get_zeros(),
        emissivity=shapes.get_zeros(),
        solar_absorptivity=shapes.get_zeros(),
        temperature1=shapes.get_zeros(),
        temperature2=shapes.get_zeros() + 1,
        resistance_at_temperature1=shapes.get_zeros(),
        resistance_at_temperature2=shapes.get_zeros(),
        aluminium_cross_section_area=1 + shapes.get_zeros(),
        constant_magnetic_effect=shapes.get_zeros(),
        current_density_proportional_magnetic_effect=shapes.get_zeros() + 1,
        max_magnetic_core_relative_resistance_increase=shapes.get_zeros(),
        steel_mass_per_unit_length=shapes.get_zeros() + 0.5,
        steel_specific_heat_capacity_at_20_celsius=shapes.get_zeros() + 500,
        steel_specific_heat_capacity_temperature_coefficient=shapes.get_zeros(),
        aluminium_mass_per_unit_length=shapes.get_zeros() + 1,
        aluminium_specific_heat_capacity_at_20_celsius=shapes.get_zeros() + 1000,
        aluminium_specific_heat_capacity_temperature_coefficient=shapes.get_zeros(),
    )
    weather = Weather(
        air_temperature=shapes.get_zeros(),
        wind_direction=shapes.get_zeros(),
        wind_speed=shapes.get_zeros(),
        ground_albedo=shapes.get_zeros(),
        clearness_ratio=shapes.get_zeros(),
    )
    span = Span(
        conductor=conductor,
        start_tower=Tower(
            shapes.get_zeros(),
            shapes.get_zeros(),
            shapes.get_zeros(),
        ),
        end_tower=Tower(
            0.1 + shapes.get_zeros(),
            0.1 + shapes.get_zeros(),
            10 + shapes.get_zeros(),
        ),
        num_conductors=1,
    )
    time = shapes.get_zeros(dtype=np.dtype("datetime64[D]"))
    model = Cigre601(span=span, weather=weather, time=time)
    final_temperature = model.compute_temperature_after_heating(
        shapes.get_zeros() + 20,
        np.timedelta64(15, "m"),
        shapes.get_zeros() + 200,
    )
    assert np.shape(final_temperature) == array_shapes.result_shape
