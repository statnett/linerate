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
To make things simpler and hopefully accurate enough, we use the same linearised model for magnetic effects in ACSR
lines as our implementation of Cigre 601.

## Solar heating

Cigre 207 assumes that the diffuse radiation is uniformly directed, this differs from Cigre 601.
Apart from this, solar heating is the same as for Cigre 601, with clearness ratio 1.

## Convective cooling

There are a few inconsistencies in the Cigre 207 modelling of convective cooling.
The definition of the Reynolds number is given as
$Re = \rho_r V D/\nu_f$, where \rho_r is the air density relative to density at sea level.
This factor is not present in the ordinary definition of Reynolds number, but seems to indicate that the kinematic
viscosity $\nu_f$ has to be corrected for the air relative density.

Cigre 207 has a non-smooth transition to low wind speeds (<0.5m/s) at angles of attack less than 45 degrees.
This leads to an increase in cooling as the wind speed drops below 0.5m/s, which seems non-physical.
This discontinuity is removed in Cigre 601.
