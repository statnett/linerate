import hypothesis
from hypothesis.extra.numpy import mutually_broadcastable_shapes
import numpy as np

from linerate.models.cigre601 import Cigre601
from linerate.types import Conductor, Span, Tower, Weather

NUMBER_OF_VARIABLES = 33


class Shapes:
    def __init__(self, shapes):
        self.shapes = shapes
        self.index = 0

    def get_zeros(self, dtype=None):
        out = np.zeros(self.shapes[self.index], dtype=dtype)
        self.index += 1
        return out


@hypothesis.given(array_shapes=mutually_broadcastable_shapes(num_shapes=NUMBER_OF_VARIABLES))
def test_vectorization(array_shapes):
    shapes = Shapes(array_shapes.input_shapes)

    conductor = Conductor(
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
        steel_mass_per_unit_length=shapes.get_zeros(),
        steel_specific_heat_capacity_at_20_celsius=shapes.get_zeros(),
        steel_specific_heat_capacity_temperature_coefficient=shapes.get_zeros(),
        aluminum_mass_per_unit_length=shapes.get_zeros(),
        aluminum_specific_heat_capacity_at_20_celsius=shapes.get_zeros(),
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
    time = shapes.get_zeros(dtype="datetime64[D]")
    model = Cigre601(span=span, weather=weather, time=time)
    final_temperature = model.compute_final_temperature(
        shapes.get_zeros() + 20,
        shapes.get_zeros(dtype="<m8[m]") + np.timedelta64(15, "m"),
        shapes.get_zeros() + 200,
    )
    assert np.shape(final_temperature) == array_shapes.result_shape
