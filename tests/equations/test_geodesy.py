import numpy as np
import pytest

from linerate.equations.geodesy import haversine_distance, bearing

# Test data adapted from pygeodesy tests
# https://github.com/mrJean1/PyGeodesy/blob/master/test/testFormy.py

Auckland = -36.8485, 174.7633
Boston = 42.3541165, -71.0693514
Cleveland = 41.499498, -81.695391
LosAngeles = 34.0522, -118.2437
MtDiablo = 37.8816, -121.9142
Newport = 41.49008, -71.312796
NewYork = 40.7791472, -73.9680804
Santiago = -33.4489, -70.6693
X = 25.2522, 55.28
Y = 14.6042, 120.982

test_distances = [
    (Boston, NewYork, 298009.404),
    (Boston, Newport, 98164.988),
    (Cleveland, NewYork, 651816.987),
    (NewYork, MtDiablo, 4084985.780),
    (Auckland, Santiago, 9670051.606),
    (Auckland, LosAngeles, 10496496.577),
    (LosAngeles, Santiago, 8998396.669),
    (X, Y, 6906867.946),
]

test_bearings = [
    (*Newport, *NewYork, 251.364),
    (*NewYork, *Santiago, 177.141),
]


@pytest.mark.parametrize("latlon1, latlon2, expected", test_distances)
def test_haversine_distance(
    latlon1: tuple[float, float], latlon2: tuple[float, float], expected: float
):
    # pygeodesy allows for up to 10% error, but 1% seems to be fine here as well
    assert haversine_distance(*latlon1, *latlon2) == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("lat1, lon1, lat2, lon2, expected", test_bearings)
def test_bearing(lat1: float, lon1: float, lat2: float, lon2: float, expected: float):
    assert bearing(lat1, lon1, lat2, lon2) == pytest.approx(np.radians(expected), rel=0.01)
