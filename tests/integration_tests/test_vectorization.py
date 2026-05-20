import hypothesis
import hypothesis.strategies as st
from hypothesis.extra.numpy import mutually_broadcastable_shapes, BroadcastableShapes
import numpy as np
import numpy.typing as npt

from linerate.models.cigre601 import Cigre601
from linerate.types import Conductor, ConductorWithTransientData, Span, Tower, Weather

NUMBER_OF_VARIABLES = 33


class Shapes:
    def __init__(self, shapes: tuple[tuple[int, ...], ...]):
        self.shapes = shapes
        self.index = 0

    def get_zeros(self, dtype: np.dtype = np.dtype("f8")) -> npt.NDArray:
        out = np.zeros(self.shapes[self.index], dtype=dtype)
        self.index += 1
        return out


@hypothesis.given(
    vectorization_indices=st.sets(st.integers(min_value=1, max_value=24), max_size=10)
)
def test_vectorization(vectorization_indices: set[int]):
    def get_zeros(index, length) -> npt.NDArray[np.floating]:
        out = np.ones(length, dtype=int)
        if index in vectorization_indices:
            out[index] = 2
        return np.zeros(out, dtype=np.dtype("f8"))

    conductor = Conductor(
        core_diameter=get_zeros(0, 25) + 1,
        conductor_diameter=get_zeros(1, 25) + 2,
        outer_layer_strand_diameter=get_zeros(2, 25),
        emissivity=get_zeros(3, 25),
        solar_absorptivity=get_zeros(4, 25),
        temperature1=get_zeros(5, 25),
        temperature2=get_zeros(6, 25) + 1,
        resistance_at_temperature1=get_zeros(7, 25),
        resistance_at_temperature2=get_zeros(8, 25),
        aluminium_cross_section_area=1 + get_zeros(9, 25),
        constant_magnetic_effect=get_zeros(10, 25),
        current_density_proportional_magnetic_effect=get_zeros(11, 25) + 1,
        max_magnetic_core_relative_resistance_increase=get_zeros(12, 25),
    )
    weather = Weather(
        air_temperature=get_zeros(13, 25),
        wind_direction=get_zeros(14, 25),
        wind_speed=get_zeros(15, 25),
        ground_albedo=get_zeros(23, 25),
        clearness_ratio=get_zeros(16, 25),
    )
    span = Span(
        conductor=conductor,
        start_tower=Tower(
            get_zeros(17, 25),
            get_zeros(18, 25),
            get_zeros(19, 25),
        ),
        end_tower=Tower(
            0.1 + get_zeros(20, 25),
            0.1 + get_zeros(21, 25),
            10 + get_zeros(22, 25),
        ),
        num_conductors=1,
    )
    time = np.array([np.datetime64("2022-01-01")]).reshape([1] * 25)
    if 24 in vectorization_indices:
        time = np.concatenate([time, time], axis=-1)

    model = Cigre601(span=span, weather=weather, time=time)
    temperature = model.compute_conductor_temperature(100)
    shape = [1] * 25
    for vectorisation_index in vectorization_indices:
        shape[vectorisation_index] = 2
    shape[0] = 1

    assert isinstance(temperature, np.ndarray)
    assert temperature.shape == tuple(shape)


@hypothesis.given(array_shapes=mutually_broadcastable_shapes(num_shapes=NUMBER_OF_VARIABLES))
def test_transient_vectorization(array_shapes: BroadcastableShapes):
    shapes = Shapes(array_shapes.input_shapes)

    conductor = ConductorWithTransientData(
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
        aluminum_mass_per_unit_length=shapes.get_zeros() + 1,
        aluminum_specific_heat_capacity_at_20_celsius=shapes.get_zeros() + 1000,
        aluminum_specific_heat_capacity_temperature_coefficient=shapes.get_zeros(),
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
        shapes.get_zeros(dtype=np.dtype("<m8[m]")) + np.timedelta64(15, "m"),
        shapes.get_zeros() + 200,
    )
    assert np.shape(final_temperature) == array_shapes.result_shape
