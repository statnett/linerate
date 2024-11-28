"""Test cases from Annex E of CIGRE TB 601."""


def test_compute_conductor_temperature(example_model_1_conductors, example_model_2_conductors):
    # Check that the ampacity of a span with two conductors is divided
    # when computing the conductor temperature.
    current_1_conductor = 1000
    current_2_conductors = current_1_conductor * 2
    # The temperature should stay the same
    assert example_model_1_conductors.compute_conductor_temperature(
        current_1_conductor
    ) == example_model_2_conductors.compute_conductor_temperature(current_2_conductors)
