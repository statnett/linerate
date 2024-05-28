from typing import Union

from linerate.units import OhmPerMeter, Ampere, SquareMeter, Unitless, SquareMeterPerAmpere


def correct_resistance_acsr_magnetic_core_loss_simple(
    ac_resistance: OhmPerMeter,
    current: Ampere,
    aluminium_cross_section_area: SquareMeter,
    constant_magnetic_effect: Union[Unitless, None],
    current_density_proportional_magnetic_effect: Union[SquareMeterPerAmpere, None],
    max_relative_increase: Unitless,
) -> OhmPerMeter:
    r"""
    Return resistance with constant correction for magnetic effects, using simple method from
    Cigre 207, see section 2.1.1.

    Parameters
    ----------
    ac_resistance:
        :math:`R~\left[\Omega\right]`. The AC resistance of the conductor.
    current:
        :math:`I~\left[\text{A}\right]`. The current going through the conductor.
    aluminium_cross_section_area:
        :math:`A_{\text{Al}}~\left[\text{m}^2\right]`. The cross sectional area of the aluminium
        strands in the conductor.
    constant_magnetic_effect:
        :math:`b`. The constant magnetic effect, most likely equal to 1. If ``None``, then no
        correction is used (useful for non-ACSR cables).
    current_density_proportional_magnetic_effect:
        :math:`m`. The current density proportional magnetic effect. If ``None``, then it is
        assumed equal to 0.
    max_relative_increase:
        :math:`c_\text{max}`. Saturation point of the relative increase in conductor resistance.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`R_\text{corrected}~\left[\Omega\right]`. The resistance of the conductor after
        taking steel core magnetization effects into account.
    """
    return 1.0123 * ac_resistance


# TODO: Implement section 2.1.2?