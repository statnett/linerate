import datetime

import hypothesis
import hypothesis.strategies as st
import numpy as np
from pytest import approx

import linerate.equations.solar_angles as solar_angles


def test_get_day_of_year_with_example():
    when1 = np.datetime64("2022-03-12T18:20")
    day1 = 71
    when2 = np.datetime64("2022-06-01T13:00")
    day2 = 152

    assert solar_angles._get_day_of_year(when1) == approx(day1)
    assert solar_angles._get_day_of_year(when2) == approx(day2)


def test_get_hour_of_day_with_example():
    when1 = np.datetime64("2022-03-12T18:20")
    hour1 = 18
    when2 = np.datetime64("2022-06-01T13:00")
    hour2 = 13

    assert solar_angles._get_hour_of_day(when1) == approx(hour1)
    assert solar_angles._get_hour_of_day(when2) == approx(hour2)


def test_get_minute_of_hour_with_example():
    when1 = np.datetime64("2022-03-12T18:20")
    minute1 = 20
    when2 = np.datetime64("2022-06-01T13:00")
    minute2 = 0

    assert solar_angles._get_minute_of_hour(when1) == approx(minute1)
    assert solar_angles._get_minute_of_hour(when2) == approx(minute2)


def test_hour_angle_relative_to_noon_with_example():
    when = np.datetime64("2022-06-01T12:00")
    omega = 0
    assert solar_angles.compute_hour_angle_relative_to_noon(when, 0) == approx(omega)


def test_hour_angle_relative_to_noon_norway():
    when = np.datetime64("2022-12-01T12:00+01:00")
    omega = 0
    longitude = 15  # Longitude at timezone +01:00
    assert solar_angles.compute_hour_angle_relative_to_noon(when, longitude) == approx(omega)


def test_solar_azimuth_variable_with_example():
    omega = 0
    Lat = 60
    delta = 0.3848280503
    assert solar_angles.compute_solar_azimuth_variable(Lat, delta, omega) == approx(0)


def test_solar_azimuth_constant_with_example():
    omega = 0
    chi = 0
    assert solar_angles.compute_solar_azimuth_constant(chi, omega) == approx(np.pi)


def test_solar_azimuth_with_example():
    C = np.pi
    chi = 0
    assert solar_angles.compute_solar_azimuth(C, chi) == approx(np.pi)


def test_solar_azimuth_with_example2():
    Lat = 50
    delta = 0.5
    omega = np.radians(25)
    chi = solar_angles.compute_solar_azimuth_variable(Lat, delta, omega)
    C = solar_angles.compute_solar_azimuth_constant(chi, np.radians(omega))
    assert chi == approx(1.231708193)
    assert C == approx(np.pi)
    assert solar_angles.compute_solar_azimuth(C, chi) == approx(4.03044563)


@hypothesis.given(degrees_of_latitude=st.floats(min_value=-180, max_value=180, allow_nan=False))
def test_sin_solar_altitude_scales_correctly_with_degrees_of_latitude(degrees_of_latitude):
    Lat = degrees_of_latitude
    delta = np.pi / 2
    omega = np.pi / 2
    H_c = np.sin(np.radians(Lat))
    assert H_c == approx(solar_angles.compute_sin_solar_altitude(Lat, delta, omega))


@hypothesis.given(solar_declination=st.floats(min_value=-23.46, max_value=23.46, allow_nan=False))
def test_sin_solar_altitude_scales_correctly_with_solar_declination(solar_declination):
    Lat = 90
    delta = solar_declination
    omega = np.pi / 2
    sin_H_c = np.sin(delta)
    assert sin_H_c == approx(solar_angles.compute_sin_solar_altitude(Lat, delta, omega))


@hypothesis.given(
    hour_angle_relative_to_noon=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False)
)
def test_sin_solar_altitude_scales_correctly_with_hour_angle_relative_to_noon(
    hour_angle_relative_to_noon,
):
    Lat = 0
    delta = 0
    omega = hour_angle_relative_to_noon
    sin_H_c = np.cos(omega)
    assert sin_H_c == approx(solar_angles.compute_sin_solar_altitude(Lat, delta, omega))


@hypothesis.given(
    day=st.dates(min_value=datetime.date(2022, 1, 1), max_value=datetime.date(2022, 12, 31))
)
def test_solar_declination_scales_correctly_with_day_of_year(day):
    day_of_year = day.timetuple().tm_yday
    N = day_of_year
    delta = np.radians((23.3) * np.sin((284 + N) * 2 * np.pi / 365))
    assert delta == approx(solar_angles.compute_solar_declination(np.datetime64(day)))


@hypothesis.given(
    when=st.datetimes(
        min_value=datetime.datetime(2022, 1, 1, 1, 0),
        max_value=datetime.datetime(2022, 12, 31, 23, 0),
    ),
    longitude=st.floats(min_value=-180, max_value=180),
)
def test_solar_declination_scales_with_dates_and_times(when, longitude):
    omega = ((-12 + when.hour + when.minute / 60 + longitude / 15) % 24) * np.pi / 12
    when = np.datetime64(when)
    assert omega == approx(solar_angles.compute_hour_angle_relative_to_noon(when, longitude))


def test_solar_declination_with_examples():
    day1 = np.datetime64("2022-06-01")
    day2 = np.datetime64("2017-10-06")
    day3 = np.datetime64("2016-03-12")
    assert solar_angles.compute_solar_declination(day1) == approx(0.382203477)
    assert solar_angles.compute_solar_declination(day2) == approx(-0.1072226616)
    assert solar_angles.compute_solar_declination(day3) == approx(-0.06275148976)


def test_sin_solar_altitude_with_example():
    Lat = 60
    omega = 0
    delta = 0.3848280503
    assert solar_angles.compute_sin_solar_altitude(Lat, delta, omega) == approx(0.7885372342)


def test_sin_solar_effective_incidence_angle_with_example():
    sin_solar_altitude = np.sin(0)
    Z_c = 1
    Z_l = 1
    assert solar_angles.compute_cos_solar_effective_incidence_angle(
        sin_solar_altitude, Z_c, Z_l
    ) == approx(1.0)


# Example testing from calculations on page 79 and 80 in CIGRE
def test_solar_declination_example_a():
    assert solar_angles.compute_solar_declination(np.datetime64("2016-06-10T11:00")) == approx(
        np.radians(22.93823993)
    )


def test_sin_solar_altitude_example_a():
    assert solar_angles.compute_sin_solar_altitude(
        30, np.radians(22.93823993), np.radians(-15)
    ) == approx(np.sin(np.radians(74.84859636)))


def test_solar_azimuth_example_a():
    chi = solar_angles.compute_solar_azimuth_variable(30, np.radians(22.93823993), np.radians(-15))
    assert chi == approx(-2.222421321)
    C = solar_angles.compute_solar_azimuth_constant(chi, np.radians(-15))
    assert C == approx(np.pi)
    # Because the two standards (IEEE738 and CIGRE601) define the solar azimuth differently, we have
    # to check (pi - IEEE_solar_azimuth == CIGRE_solar_azimuth). That is why the np.pi is added.
    assert (np.pi - solar_angles.compute_solar_azimuth(C, chi)) == approx(1.147975923)


def test_compute_sin_solar_altitude_for_span(example_span_1_conductor):
    time = np.arange(
        np.datetime64("2023-06-10T00:00"), np.datetime64("2023-06-11T00:00"), np.timedelta64(1, "h")
    )
    sin_H_s = solar_angles.compute_sin_solar_altitude_for_span(example_span_1_conductor, time)

    assert sin_H_s.shape == time.shape

    assert sin_H_s == approx(
        [
            -0.6037028727577956,
            -0.5765123992400065,
            -0.4967939645224945,
            -0.3699802481903158,
            -0.204713395441933,
            -0.012256069175761386,
            0.1942760818739301,
            0.4008082329236209,
            0.5932655591897926,
            0.7585324119381758,
            0.8853461282703547,
            0.9650645629878669,
            0.9922550365056562,
            0.9650645629878669,
            0.885346128270355,
            0.758532411938176,
            0.5932655591897933,
            0.40080823292362144,
            0.1942760818739303,
            -0.012256069175760664,
            -0.2047133954419325,
            -0.36998024819031544,
            -0.4967939645224943,
            -0.5765123992400065,
        ]
    )


def test_compute_sin_solar_effective_incidence_angle_for_span(example_span_1_conductor):
    time = np.arange(
        np.datetime64("2023-06-10T00:00"), np.datetime64("2023-06-11T00:00"), np.timedelta64(1, "h")
    )
    sin_H_s = solar_angles.compute_sin_solar_altitude_for_span(example_span_1_conductor, time)

    sin_eta = solar_angles.compute_sin_solar_effective_incidence_angle_for_span(
        example_span_1_conductor, time, sin_H_s
    )

    assert sin_eta.shape == time.shape

    assert sin_eta == approx(
        [
            1,
            0.9711468921150654,
            0.8875493154679516,
            0.7586086903559678,
            0.6026859691738063,
            0.4559029843231809,
            0.3885528601139044,
            0.45590257803393736,
            0.6026854368482859,
            0.7586082020187381,
            0.8875489539951258,
            0.9711467013836234,
            1,
            0.9711467478944035,
            0.8875490106123987,
            0.7586081859456092,
            0.6026851915732289,
            0.4559018377845522,
            0.38855146738201046,
            0.4559014314940192,
            0.6026846592466302,
            0.7586076976076148,
            0.8875486491389878,
            0.9711465571624637,
        ]
    )
