import numpy as np

from linerate.units import Degrees, Meter, Radian


def bearing(
    latitude_1: Degrees, longitude_1: Degrees, latitude_2: Degrees, longitude_2: Degrees
) -> Radian:
    r""":math:`\gamma_c~\left[\text{radian}\right]`. Angle (east of north) from a pair of lat-lon coordinates.

    Parameters
    ----------
    latitude_1:
        :math:`\phi_1~\left[^\circ\right]`. First latitude.
    longitude_1:
        :math:`\phi_1~\left[^\circ\right]`. First longitude (east of the prime meridian).
    latitude_2:
        :math:`\phi_2~\left[^\circ\right]`. Second latitude.
    longitude_2:
        :math:`\phi_2~\left[^\circ\right]`. Second longitude (east of the prime meridian).

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        Azimuth between two points.
    """
    # Adapted from https://www.movable-type.co.uk/scripts/latlong.html
    phi_1 = np.radians(latitude_1)
    phi_2 = np.radians(latitude_2)
    delta_lambda = np.radians(longitude_2 - longitude_1)
    y = np.sin(delta_lambda) * np.cos(phi_2)
    x = np.cos(phi_1) * np.sin(phi_2) - np.sin(phi_1) * np.cos(phi_2) * np.cos(delta_lambda)
    return (np.atan2(y, x) + 2 * np.pi) % (2 * np.pi)  # Change from (-pi, pi) to (0, 2*pi)


def haversine_distance(
    latitude_1: Degrees, longitude_1: Degrees, latitude_2: Degrees, longitude_2: Degrees
) -> Meter:
    r""":math:`\left[\text{m}\right]`. Distance between two points assuming spherical earth.

    Parameters
    ----------
    latitude_1:
        :math:`\phi_1~\left[^\circ\right]`. First latitude.
    longitude_1:
        :math:`\phi_1~\left[^\circ\right]`. First longitude (east of the prime meridian).
    latitude_2:
        :math:`\phi_2~\left[^\circ\right]`. Second latitude.
    longitude_2:
        :math:`\phi_2~\left[^\circ\right]`. Second longitude (east of the prime meridian).

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`d~\left[\text{m}\right]`. Distance between two points.
    """
    # Adapted from https://www.movable-type.co.uk/scripts/latlong.html
    R = 6371008.771415  # Earth radius
    phi_1 = np.radians(latitude_1)
    phi_2 = np.radians(latitude_2)
    lambda_1 = np.radians(longitude_1)
    lambda_2 = np.radians(longitude_2)
    delta_phi = phi_2 - phi_1
    delta_lambda = lambda_2 - lambda_1
    hav_theta = (
        np.sin(delta_phi / 2) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
    )
    theta = 2 * np.atan2(np.sqrt(hav_theta), np.sqrt(1 - hav_theta))
    return theta * R
