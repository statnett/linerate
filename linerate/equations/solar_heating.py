from linerate.units import Meter, Unitless, WattPerMeter, WattPerSquareMeter


def compute_solar_heating(
    absorptivity: Unitless,
    global_radiation_intensity: WattPerSquareMeter,
    conductor_diameter: Meter,
) -> WattPerMeter:
    r"""Compute the solar heating experienced by the conductor.

    Equation (8) on page 18 of :cite:p:`cigre601` and (11) on page 4 in :cite:p:`cigre207`.

    Parameters
    ----------
    absorptivity:
        :math:`\alpha_s`. Material constant. According to :cite:p:`cigre601`, it starts at
        approximately 0.2 for new cables and reaches a constant value of approximately 0.9
        after about one year.
    global_radiation_intensity:
        :math:`I_T~\left[\text{W}~\text{m}^{-2}\right]`.The global radiation intensity.
    conductor_diameter:
        :math:`D~\left[\text{m}\right]`. Outer diameter of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`P_S~\left[\text{W}~\text{m}^{-1}\right]`. The solar heating of the conductor
    """
    alpha_s = absorptivity
    I_T = global_radiation_intensity
    D = conductor_diameter

    return alpha_s * I_T * D
