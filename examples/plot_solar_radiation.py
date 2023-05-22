r"""
Exploration of heating from solar radiation
-------------------------------------------

In this example, we see how the solar radiation varies with the solar altitude angle, :math:`H_s`,
the difference between the solar azimuth and the span azimuth (or bearing),
:math:`\left|\gamma_c - \gamma_s\right|`, and the albedo, :math:`F`.
"""

###############################################################################
# Imports and utilities
# ^^^^^^^^^^^^^^^^^^^^^

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from linerate.equations import cigre601

###############################################################################
# Simulation parameters
# ^^^^^^^^^^^^^^^^^^^^^

solar_altitude = np.linspace(0, np.pi / 2)[:, np.newaxis]
azimuth_difference = np.linspace(0, np.pi / 2, 10)[np.newaxis, :]
albedo = 0.2
height_above_sea_level = 0

###############################################################################
# Intermediate calculations
# ^^^^^^^^^^^^^^^^^^^^^^^^^

sin_H_s = np.sin(solar_altitude)
cos_eta = np.cos(solar_altitude) * np.cos(azimuth_difference)
sin_eta = np.sqrt(1 - cos_eta**2)

###############################################################################
# Compute the solar radiation components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

I_B = cigre601.solar_heating.compute_direct_solar_radiation(
    sin_H_s, clearness_ratio=1, height_above_sea_level=height_above_sea_level
)
I_d = cigre601.solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)
I_T_F0 = cigre601.solar_heating.compute_global_radiation_intensity(
    I_B, I_d, albedo=0.0, sin_angle_of_sun_on_line=sin_eta, sin_solar_altitude=sin_H_s
)
I_T_F = cigre601.solar_heating.compute_global_radiation_intensity(
    I_B, I_d, albedo=albedo, sin_angle_of_sun_on_line=sin_eta, sin_solar_altitude=sin_H_s
)

###############################################################################
# Create visualisation
# ^^^^^^^^^^^^^^^^^^^^
#
# This section can be skipped, since most of the code is just there to create pretty plots.

# Setup figure and axes
fig = plt.figure(figsize=(11, 1.9))
axes = [fig.add_axes([0.07, 0.27, 0.19, 0.7])]
axes += [
    fig.add_axes([0.30, 0.27, 0.18, 0.7], sharey=axes[0]),
    fig.add_axes([0.53, 0.27, 0.18, 0.7], sharey=axes[0]),
    fig.add_axes([0.76, 0.27, 0.18, 0.7], sharey=axes[0]),
]
cbar_ax = fig.add_axes([0.955, 0.27, 0.015, 0.7])

# Add plots
axes[0].plot(np.degrees(solar_altitude), I_B, color="k")
axes[1].plot(np.degrees(solar_altitude), I_d, color="k")
for i, d_gamma in enumerate(azimuth_difference.ravel()):
    color = cm.cividis(d_gamma / azimuth_difference.max())
    d_gamma = np.degrees(d_gamma)
    axes[2].plot(np.degrees(solar_altitude), I_T_F0[:, i], color=color)
    axes[3].plot(np.degrees(solar_altitude), I_T_F[:, i], color=color)


for ax in axes:
    # Setup y-axes
    ax.set_xlabel(r"Solar altitude $[^\circ]$")
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 30, 60, 90])

    # Setup y-axes to be shared
    ax.set_ylim(0, I_T_F.max() * 1.05)
    ax.set_yticks([0, 500, 1000, 1360])  # Include tick for solar constant
    ax.set_yticklabels([0, 500, 1000, "$G_{SC}$"])
    ax.axhline(1360, color="k", linestyle="--")  # Add dashed line for solar constant


# Removed shared ticks
axes[1].tick_params(labelleft=False)
axes[2].tick_params(labelleft=False)
axes[3].tick_params(labelleft=False)

# Setup labels
axes[0].set_ylabel(r"$I_B~[\mathrm{W}~\mathrm{m}^{-1}]$", labelpad=-1)
axes[1].set_ylabel(r"$I_d~[\mathrm{W}~\mathrm{m}^{-1}]$")
axes[2].set_ylabel("$F=0$\n$I_T~[\mathrm{W}~\mathrm{m}^{-1}]$")  # noqa
axes[3].set_ylabel(f"$F={albedo}$\n$I_T~[\mathrm{{W}}~\mathrm{{m}}^{{-1}}]$")  # noqa

# Colorbar
cbar_ax.imshow(azimuth_difference.T, aspect="auto", cmap="cividis")
cbar_ax.yaxis.set_label_position("right")
cbar_ax.set_ylabel(r"$\left|\gamma_c - \gamma_s\right|~[^\circ]$", labelpad=-10)
cbar_ax.yaxis.tick_right()
cbar_ax.set_yticks(cbar_ax.get_ylim())
cbar_ax.set_yticklabels([0, 90])
cbar_ax.set_xticks([])

plt.show()
