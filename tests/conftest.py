import hypothesis
import pytest

hypothesis.settings.register_profile("default", deadline=None)
hypothesis.settings.load_profile("default")


@pytest.fixture
def random_seed(pytestconfig):
    return pytestconfig.getoption("randomly_seed")


@pytest.fixture
def rng(random_seed):
    import numpy as np

    return np.random.default_rng(random_seed)
