import numpy as np

import linerate


def test_vectorization():
    def get_zeros(index, length):
        out = np.ones(length, dtype=int)
        out[index] = 2
        return np.zeros(out)

    conductor = linerate.types.Conductor(
        core_diameter=1,
        conductor_diameter=1,
        outer_layer_strand_diameter=0.1,
        emissivity=get_zeros(0, 16),
        solar_absorptivity=get_zeros(1, 16),
        temperature1=get_zeros(2, 16),
        temperature2=get_zeros(3, 16)+1,
        resistance_at_temperature1=get_zeros(4, 16),
        resistance_at_temperature2=get_zeros(5, 16),
        aluminium_cross_section_area=1 + get_zeros(6, 16),
        constant_magnetic_effect=get_zeros(7, 16),
        current_density_proportional_magnetic_effect=get_zeros(8, 16) + 1,
        max_magnetic_core_relative_resistance_increase=get_zeros(9, 16),
    )
    weather = linerate.types.Weather(
        air_temperature=get_zeros(10, 16),
        wind_direction=get_zeros(11, 16),
        wind_speed=get_zeros(12, 16),
        clearness_ratio=get_zeros(13, 16),
    )
    span = linerate.Span(
        conductor=conductor,
        start_tower=linerate.Tower(0, 0, 0),
        end_tower=linerate.Tower(0.1, 0.1, 10),
        ground_albedo=get_zeros(14, 16),
        num_conductors=1,
    )
    time = np.array([np.datetime64("2022-01-01")] * 2).reshape([1] * 15 + [2])

    model = linerate.model.Cigre601(span=span, weather=weather, time=time)
    temperature = model.compute_conductor_temperature(100)
    assert temperature.shape == tuple([2] * 16)
