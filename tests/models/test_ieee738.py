import numpy as np

from linerate.models.ieee738 import IEEE738
from linerate.types import Span, Weather


def test_global_radiation_intensity_example(
    example_span_1_conductor: Span, example_weather_a: Weather
):
    when = np.datetime64("2022-03-12T12:20")
    model = IEEE738(example_span_1_conductor, example_weather_a, when)
    assert model.compute_global_radiation_intensity() == 984.1125892531261
