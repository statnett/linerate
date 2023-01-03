import hypothesis
import hypothesis.strategies as st
import numpy as np

import linerate


@hypothesis.given(
    vectorization_indices=st.sets(st.integers(min_value=1, max_value=24), max_size=10)
)
def test_vectorization(vectorization_indices):
    def get_zeros(index, length):
        out = np.ones(length, dtype=int)
        if index in vectorization_indices:
            out[index] = 2
        return np.zeros(out)

    conductor = linerate.types.Conductor(
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
    weather = linerate.types.Weather(
        air_temperature=get_zeros(13, 25),
        wind_direction=get_zeros(14, 25),
        wind_speed=get_zeros(15, 25),
        clearness_ratio=get_zeros(16, 25),
    )
    span = linerate.Span(
        conductor=conductor,
        start_tower=linerate.Tower(
            get_zeros(17, 25),
            get_zeros(18, 25),
            get_zeros(19, 25),
        ),
        end_tower=linerate.Tower(
            0.1 + get_zeros(20, 25),
            0.1 + get_zeros(21, 25),
            10 + get_zeros(22, 25),
        ),
        ground_albedo=get_zeros(23, 25),
        num_conductors=1,
    )
    time = np.array([np.datetime64("2022-01-01")]).reshape([1] * 25)
    if 24 in vectorization_indices:
        time = np.concatenate([time, time], axis=-1)

    model = linerate.model.Cigre601(span=span, weather=weather, time=time)
    temperature = model.compute_conductor_temperature(100)
    shape = [1] * 25
    for vectorisation_index in vectorization_indices:
        shape[vectorisation_index] = 2
    shape[0] = 1
    assert temperature.shape == tuple(shape)
