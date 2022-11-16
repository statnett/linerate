import numpy as np

import linerate


def test_vectorization():
    def get_zeros(index, length):
        out = np.ones(length, dtype=int)
        out[index] = 2
        return np.zeros(out)

    conductor = linerate.types.Conductor(
        core_diameter=get_zeros(0, 19) + 1,
        conductor_diameter=get_zeros(1, 19) + 2,
        outer_layer_strand_diameter=get_zeros(2, 19),
        emissivity=get_zeros(3, 19),
        solar_absorptivity=get_zeros(4, 19),
        temperature1=get_zeros(5, 19),
        temperature2=get_zeros(6, 19) + 1,
        resistance_at_temperature1=get_zeros(7, 19),
        resistance_at_temperature2=get_zeros(8, 19),
        aluminium_cross_section_area=1 + get_zeros(9, 19),
        constant_magnetic_effect=get_zeros(10, 19),
        current_density_proportional_magnetic_effect=get_zeros(11, 19) + 1,
        max_magnetic_core_relative_resistance_increase=get_zeros(12, 19),
    )
    weather = linerate.types.Weather(
        air_temperature=get_zeros(13, 19),
        wind_direction=get_zeros(14, 19),
        wind_speed=get_zeros(15, 19),
        clearness_ratio=get_zeros(16, 19),
    )
    span = linerate.Span(
        conductor=conductor,
        start_tower=linerate.Tower(0, 0, 0),
        end_tower=linerate.Tower(0.1, 0.1, 10),
        ground_albedo=get_zeros(17, 19),
        num_conductors=1,
    )
    time = np.array([np.datetime64("2022-01-01")] * 2).reshape([1] * 18 + [2])

    model = linerate.model.Cigre601(span=span, weather=weather, time=time)
    temperature = model.compute_conductor_temperature(100)
    assert temperature.shape == tuple([1] + [2] * 18)
