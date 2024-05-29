# Cigre 207

## Cigre 207 vs 601

The Cigre 601 report states that 601 takes into account improvements in the calculation of AC resistance of stranded
conductors, includes a more realistic estimation of the axial and radial temperature distributions, more coherent
convection model for low wind speeds, and has a more flexible solar radiation model.

## Skin effect

$P_J = k_j I^2 R_{dc}$
Where $k_j$ is the skin effect factor.
We do not implement skin effect, since this is included in the AC resistance values we use as input.

## Magnetic effect

The effect of magnetic heating seems to be intertwined with the skin effect in Cigre 207.
To make things simpler and hopefully accurate enough, we use the same linearised model as our implementation of
Cigre 601.
**Maybe we have to come back to this one...**

## Solar heating

Solar heating is the same as for Cigre 601, but with clearness ratio 1.

## Convective cooling
There are a few inconsistencies in the Cigre 207 modelling of convective cooling.
The definition of the Reynolds number is given as
$Re = \rho_r V D/\nu_f$, where \rho_r is the air density relative to density at sea level.
This factor is not present in the ordinary definition of Reynolds number, so we choose to omit it.