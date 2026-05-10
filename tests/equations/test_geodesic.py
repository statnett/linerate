import pytest

from linerate.equations.geodesic import haversine_distance


def test_haversine_distance():
    # Adapted from wikipedia haversine distance example with accuracy of one kilometer.
    lat1, lon1, lat2, lon2 = 38.898, -77.037, 48.858, 2.294
    assert haversine_distance(lat1, lon1, lat2, lon2) == pytest.approx(6161000, abs=1000)
