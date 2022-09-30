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
        emissivity=get_zeros(0, 15),
        solar_absorptivity=get_zeros(1, 15),
        resistance_at_20c=get_zeros(2, 15),
        linear_resistance_coefficient_20c=get_zeros(3, 15),
        quadratic_resistance_coefficient_20c=get_zeros(4, 15),
        aluminium_surface_area=1 + get_zeros(5, 15),
        constant_magnetic_effect=get_zeros(6, 15),
        current_density_proportional_magnetic_effect=get_zeros(7, 15) + 1,
        max_magnetic_core_relative_resistance_increase=get_zeros(8, 15),
    )
    weather = linerate.types.Weather(
        air_temperature=get_zeros(9, 15),
        wind_direction=get_zeros(10, 15),
        wind_speed=get_zeros(11, 15),
        clearness_ratio=get_zeros(12, 15),
    )
    span = linerate.Span(
        conductor=conductor,
        start_tower=linerate.Tower(0, 0, 0),
        end_tower=linerate.Tower(0.1, 0.1, 10),
        ground_albedo=get_zeros(13, 15),
        num_conductors=1,
    )
    time = np.array([np.datetime64("2022-01-01")] * 2).reshape([1] * 14 + [2])

    model = linerate.model.Cigre601(span=span, weather=weather, time=time)
    temperature = model.compute_conductor_temperature(100)
    assert temperature.shape == tuple([2] * 15)
