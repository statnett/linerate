from textwrap import dedent

import numpy as np

from linerate.units import (
    Celsius,
    JoulePerKilogramPerKelvin,
    KilogramPerMeterPerSecond,
    Meter,
    MeterPerSecond,
    MeterPerSquareSecond,
    SquareMeterPerSecond,
    Unitless,
    WattPerMeterPerKelvin,
)


def compute_reynolds_number(
    wind_speed: MeterPerSecond,
    conductor_diameter: Meter,
    kinematic_viscosity_of_air: SquareMeterPerSecond,
) -> Unitless:
    r"""Compute the Reynolds number using the conductor diameter as characteristic length scale.

    Defined in the text on page 25 of :cite:p:`cigre601` and equation (2c)
    on page 10 in :cite:p:`ieee738`.

    The Reynolds number is a dimensionless quantity that can be used to assess if a stream is
    likely to be turbulent or not. It is given by

    .. math::

        \text{Re} = \frac{v L}{\nu},

    where :math:`v` is the flow velocity, :math:`L` is a *characteristic length* (in our case,
    the conductor diameter) and :math:`\nu` is the kinematic viscosity.

    Parameters
    ----------
    wind_speed:
        :math:`v~\left[\text{m}~\text{s}^{-1}\right]`. The wind speed.
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    kinematic_viscosity_of_air:
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Re}`. The Reynolds number.
    """
    v = wind_speed
    D = conductor_diameter
    nu_f = kinematic_viscosity_of_air
    return v * D / nu_f


def compute_grashof_number(
    conductor_diameter: Meter,
    surface_temperature: Celsius,
    air_temperature: Celsius,
    kinematic_viscosity_of_air: SquareMeterPerSecond,
    coefficient_of_gravity: MeterPerSquareSecond = 9.807,
) -> Unitless:
    r"""Compute the Grashof number.

    Defined in the nomenclature on page 7 of :cite:p:`cigre601`
    and on page 5 of :cite:p:`cigre207`.

    The Grashof number is a dimensionless quantity that can be used to assess the degree of free
    and forced convective heat transfer.

    Parameters
    ----------
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    surface_temperature:
        :math:`T_s~\left[^\circ\text{C}\right]`. The conductor surface temperature.
    air_temperature:
        :math:`T_a~\left[^\circ\text{C}\right]`. The ambient air temperature.
    kinematic_viscosity_of_air:
        :math:`\nu_f~\left[\text{m}^2~\text{s}^{-1}\right]`. The kinematic viscosity of air.
    coefficient_of_gravity:
        :math:`g~\left[\text{m}~\text{s}^{-2}\right]`. The graviatational constant, optional
        (default=9.807).

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Gr}`. The Grashof number.
    """
    T_s = surface_temperature
    T_a = air_temperature
    T_f = 0.5 * (T_s + T_a)
    D = conductor_diameter
    nu_f = kinematic_viscosity_of_air
    g = coefficient_of_gravity

    return (D**3) * np.abs((T_s - T_a)) * g / ((T_f + 273.15) * (nu_f**2))


def compute_prandtl_number(
    thermal_conductivity_of_air: WattPerMeterPerKelvin,
    dynamic_viscosity_of_air: KilogramPerMeterPerSecond,
    specific_heat_capacity_of_air: JoulePerKilogramPerKelvin,
) -> Unitless:
    r"""Compute the Prandtl number.

    Defined in the nomenclature on page 8 of :cite:p:`cigre601`.

    The Prandtl number measures the ratio between viscosity and thermal diffusivity for a fluid.

    Parameters
    ----------
    thermal_conductivity_of_air:
        :math:`\lambda_f~\left[\text{W}~\text{m}^{-1}~\text{K}^{-1}\right]`. The thermal
        conductivity of air at the given temperature.
    dynamic_viscosity_of_air:
        :math:`\mu_f~\left[\text{kg}~\text{m}^{-1}~\text{s}^{-1}\right]`. The dynamic viscosity of
        air.
    specific_heat_capacity_of_air:
        :math:`\text{J}~\left[\text{kg}^{-1}~\text{K}^{-1}\right]`. The specific heat capacity of
        air.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Pr}`. The Prandtl number.
    """
    lambda_f = thermal_conductivity_of_air
    mu_f = dynamic_viscosity_of_air
    c_f = specific_heat_capacity_of_air

    return c_f * mu_f / lambda_f


def compute_conductor_roughness(
    conductor_diameter: Meter,
    outer_layer_strand_diameter: Meter,
) -> Unitless:
    r"""Compute the surface roughness of the conductor.

    Defined in the text on page 25 of :cite:p:`cigre601` and on page 6 of :cite:p:`cigre207`.

    Parameters
    ----------
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.
    outer_layer_strand_diameter:
        :math:`d~\left[\text{m}\right]`. The diameter of the strands in the outer layer of the
        conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`\text{Rs}`. The roughness number
    """
    D = conductor_diameter
    d = outer_layer_strand_diameter

    if np.any(d < 0):
        raise ValueError(
            dedent(
                """\
            Cannot have negative outer layer strand diameter. If it was set this way to signify
            that it is a smooth conductor, then set the outer layer strand diameter to nan or zero
            instead.\
            """
            )
        )
    if np.any(d >= D):
        raise ValueError(
            "The outer layer strand diameter must be strictly smaller than the conductor diameter."
        )

    return d / (2 * (D - d))
