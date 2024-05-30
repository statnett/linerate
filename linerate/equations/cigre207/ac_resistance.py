from linerate.units import OhmPerMeter


def correct_resistance_for_skin_effect(
    dc_resistance: OhmPerMeter,
) -> OhmPerMeter:
    r"""
    Return resistance with constant correction for skin effect, using simple method from
    Cigre 207, see section 2.1.1.

    Parameters
    ----------
    dc_resistance:
        :math:`R~\left[\Omega\right]`. The DC resistance of the conductor.

    Returns
    -------
    Union[float, float64, ndarray[Any, dtype[float64]]]
        :math:`R_\text{corrected}~\left[\Omega\right]`. The resistance of the conductor after
        taking skin effect into account.
    """
    return 1.0123 * dc_resistance


# TODO: Implement section 2.1.2?
