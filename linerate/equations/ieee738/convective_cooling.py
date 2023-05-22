import numpy as np

from linerate.units import (
    Celsius,
    KilogramPerCubeMeter,
    KilogramPerMeterPerSecond,
    Meter,
    MeterPerSecond,
    Radian,
    SquareMeterPerSecond,
    Unitless,
    WattPerMeter,
    WattPerMeterPerCelsius,
    WattPerMeterPerKelvin,
)


def compute_air_temperature_at_boundary_layer(  # T_film
    temperature_of_conductor_surface: Celsius,
    temperature_of_ambient_air: Celsius,
) -> Celsius:
    r"""Compute the temperature at the boundary layer, which is the thin air-film surrounding
    the conductor. Equal to the average of the ambient air temperature and the conductor
    surface temperature.

    Equation (6) on page 12 of :cite:p:`ieee738`.

    Parameters
    ----------
    temperature_of_conductor_surface:
        :math:'T_s ~left[\circ\text{C}\right]`. The temperature of the surface of the conductor.
    temperature_of_ambient_air:
        :math:'T_a ~left[\circ\text{C}\right]`. The temperature of the ambient air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`(T_s + T_s)/2~\left[^\circ\text{C}\right]`. The temperature at the boundary layer,
        the thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor surface temperature.
    """
    T_s = temperature_of_conductor_surface
    T_a = temperature_of_ambient_air
    return (T_s + T_a) / 2


def compute_dynamic_viscosity_of_air(  # mu_f
    air_temperature_at_boundary_layer: Celsius,
) -> KilogramPerMeterPerSecond:
    r"""Approximation of the dynamic viscosity of air at a given temperature.

    Equation (13a) on page 17 of :cite:p:`ieee738`.

    Parameters
    ----------
    film_temperature:
        :math:`(T_s + T_s)/2~\left[^\circ\text{C}\right]`. The temperature at the boundary layer,
        the thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor surface temperature.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\mu_f~\left[\text{kg}~\text{m}^{-1}~\text{s}^{-1}\right]`. The dynamic viscosity
        of air.
    """
    T_film = air_temperature_at_boundary_layer
    return (1.458e-6 * (T_film + 273) ** 1.5) / (T_film + 383.4)


def compute_kinematic_viscosity_of_air(  # nu_f
    dynamic_viscosity_of_air: KilogramPerMeterPerSecond, air_density: KilogramPerCubeMeter
) -> SquareMeterPerSecond:
    r"""Compute the kinematic viscosity of air.

    Definition in text on page 25 of :cite:p:`cigre601`.

    Parameters
    ----------
    dynamic_viscosity_of_air:
        :math:`\mu_f~\left[\text{kg}~\text{m}^{-1}~\text{s}^{-1}\right]`. The dynamic viscosity of
        air.
    air_density:
        :math:`\rho_f~\left[\text{kg}~\text{m}^{-3}\right]`. The mass density of air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.
    """
    return dynamic_viscosity_of_air / air_density


def compute_reynolds_number(
    wind_speed: MeterPerSecond,
    conductor_diameter: Meter,
    kinematic_viscosity_of_air: SquareMeterPerSecond,
) -> Unitless:
    r"""Compute the Reynolds number using the conductor diameter as characteristic length scale.

    Equation (2c) on page 10 in :cite:p:`ieee738`

    Separated out kinematic viscosity of air into its own function because that is how CIGRE601
    has done it, and that way they do the calculation the same way. Kinematic viscosity is given by

    .. math:

        \nu_f = \frac{\mu_f}{\rho_f}

    where {\mu_f} is the dynamic viscosity of air and {\rho_f} is the air density.

    The Reynolds number is a dimensionless quantity that can be used to assess if a stream is
    likely to be turbulent or not. It is given by

    .. math::

        \text{N_{Re}} = \frac{V_w D_0}{\nu_f},

    where :math:`V_w` is the flow velocity, :math:`D_0` is a *characteristic length* (in our case,
    the conductor diameter) and :math:`\nu_f` is the kinematic viscosity.

    Parameters
    ----------
    wind_speed:
        :math:`V_w~\left[\text{m}~\text{s}^{-1}\right]`. The wind speed.
    conductor_diameter:
        :math:`D_0~\left[\text{m}\right]`. Outer diameter of the conductor.
    kinematic_viscosity_of_air:
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.

    """
    V_w = wind_speed
    D_0 = conductor_diameter
    nu_f = kinematic_viscosity_of_air
    return V_w * D_0 / nu_f


def compute_wind_direction_factor(  # K_angle
    angle_of_attack: Radian,
) -> Unitless:
    r"""Compute the wind direction factor.

    Equation (4a) on page 11 of :cite:p:`ieee738`.

    The wind direction factor is used to calculate forced convection.

    This angle is called Phi in CIGRE601.

    Parameters
    ----------
    angle_of_attack:
        :math:`\phi~\left[\text{radian}\right]`. The wind angle-of-attack.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math: '\text{K_{angle}'. The wind direction factor.

    """
    Phi = angle_of_attack
    return 1.194 - np.cos(Phi) + 0.194 * np.cos(2 * Phi) + 0.368 * np.sin(2 * Phi)


def compute_thermal_conductivity_of_air(  # k_f
    air_temperature_at_boundary_layer: Celsius,
) -> WattPerMeterPerCelsius:
    r"""Approximation of the thermal conductivity of air.

    Equation (15a) on page 18 of :cite:p:`ieee738`.

    Parameters
    ----------
    film_temperature:
        :math:`(T_s + T_s)/2~\left[^\circ\text{C}\right]`. The temperature at the boundary layer,
        the thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor surface temperature.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{k}_{f}~\left[\text{W}~\text{m}^{-1}~^\circ\text{C}^{-1}\right]`. The thermal
        conductivity of air at the boundary layer temperature.

    """
    T_film = air_temperature_at_boundary_layer
    A = 2.424e-2
    B = 7.477e-5
    C = -4.407e-9
    return np.poly1d([C, B, A])(T_film)


def compute_forced_convection(  # q_c1 or q_c2
    wind_direction_factor: Radian,
    reynolds_number: Unitless,
    thermal_conductivity_of_air: WattPerMeterPerKelvin,
    temperature_of_conductor_surface: Celsius,
    temperature_of_ambient_air: Celsius,
) -> WattPerMeter:
    r"""Compute the forced convection.

    Equation (3a) and (3b) on page 11 of :cite:p:`ieee738`.

    According to :cite:p:`ieee738`, "Equation (3a) is correct at low winds but underestimates
    forced convection at high wind speeds. Equation (3b) is correct at high wind speeds but
    underestimates forced convection at low wind speeds. At any wind speed, this standard recommends
    calculating convective heat loss with both equations, and using the larger of the two calculated
    convection heat loss rates."

    Parameters
    ----------
    wind_direction_factor:
        :math: '\text{K_{angle}'. The wind direction factor
    reynolds_number:
        :math:`\text{N_{Re}}`. The Reynolds number.
    thermal_conductivity_of_air:
        :math:`\text{k_f}~\left[\text{W}~\text{m}^{-1}~^\circ\text{C}^{-1}\right]`. The thermal
        conductivity of air at the boundary layer temperature.
    temperature_of_conductor_surface:
        :math:'T_s ~left[\circ\text{C}\right]`. The temperature of the surface of the conductor.
    temperature_of_ambient_air:
        :math:'T_a ~left[\circ\text{C}\right]`. The temperature of the ambient air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{q}_{c}`. The forced convection.

    """
    K_angle = wind_direction_factor
    N_Re = reynolds_number
    k_f = thermal_conductivity_of_air
    T_s = temperature_of_conductor_surface
    T_a = temperature_of_ambient_air

    q_c1 = K_angle * (1.01 + 1.35 * N_Re**0.52) * k_f * (T_s - T_a)
    q_c2 = K_angle * 0.754 * N_Re**0.6 * k_f * (T_s - T_a)
    # if q_c1 > q_c2:
    #     return q_c1
    # return q_c2

    if hasattr(q_c1, "__len__"):
        q_cf = []
        for i in range(len(q_c1)):
            if q_c1[i] > q_c2[i]:
                q_cf.append(q_c1[i])
            else:
                q_cf.append(q_c2[i])
        return np.array(q_cf)
    else:
        if q_c1 > q_c2:
            return q_c1
        return q_c2


def compute_air_density(  # rho_f
    air_temperature_at_boundary_layer: Celsius,
    elevation: Meter,
) -> KilogramPerCubeMeter:
    r"""Compute the air density.

    Equation (14a) on page 17 of :cite:p:`ieee738`.

    The air density at the elevation of the conductor at the temperature at the boundary layer.

    Parameters
    ----------
    film_temperature:
        :math:`(T_s + T_s)/2~\left[^\circ\text{C}\right]`. The temperature at the boundary layer,
        the thin air-film surrounding the conductor. Equal to the average of the ambient air
        temperature and the conductor surface temperature.
    elevation:
        :math:`H_e~\left[\text{m}\right]`. The elevation of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\rho_{f}`. The air density.

    """
    T_film = air_temperature_at_boundary_layer
    H_e = elevation
    return (np.poly1d([6.379e-9, -1.525e-4, 1.293])(H_e)) / (1 + 0.00367 * T_film)


def compute_natural_convection(  # q_cn
    air_density: KilogramPerCubeMeter,
    conductor_diameter: Meter,
    temperature_of_conductor_surface: Celsius,
    temperature_of_ambient_air: Celsius,
) -> WattPerMeter:
    r"""Compute the natural convection.

    Equation (5a) on page 12 of :cite:p:`ieee738`.

    Temperature of conductor surface must be larger than or equal to temperature of ambient air

    Parameters
    ----------
    air_density:
        :math:`\rho_{f}`. The air density.
    conductor_diameter:
        :math:`D_0~\left[\text{m}\right]`. Outer diameter of the conductor.
    temperature_of_conductor_surface:
        :math:'T_s ~left[\circ\text{C}\right]`. The temperature of the surface of the conductor.
    temperature_of_ambient_air:
        :math:'T_a ~left[\circ\text{C}\right]`. The temperature of the ambient air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`q_{cn}~\left[\text{W}~\text{m}^{-1}\right]`. The natural convection.
    """
    rho_f = air_density
    D_0 = conductor_diameter
    T_s = temperature_of_conductor_surface
    T_a = temperature_of_ambient_air
    return 3.645 * rho_f**0.5 * D_0**0.75 * (T_s - T_a) ** 1.25


def compute_convective_cooling(
    forced_convection: WattPerMeter,
    natural_convection: WattPerMeter,
) -> WattPerMeter:
    r"""Compute the convective cooling of the conductor.

    On page 11 in :cite:p:`ieee738, it says that one should calculate both forced and natural
    convection, and choose the larger of the two as the convective cooling.

    Parameters
    ----------
    forced_convection:
        :math:`q_c. The forced convection.
    natural_convection:
        :math:`q_{cn}~\left[\text{W}~\text{m}^{-1}\right]`. The natural convection.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`q_c~\left[\text{W}~\text{m}^{-1}\right]`. The convective cooling of the conductor.
        Either equal to the forced or the natural convection, whichever is the largest.
    """
    q_cf = forced_convection
    q_cn = natural_convection

    return np.maximum(q_cf, q_cn)
