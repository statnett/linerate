r"""
Comparing the methods for calculating solar heating in CIGRE601 and IEEE738
--------------------------------------------------------------------------------

In this example, we see how the solar heating varies with different variables.
This calculation is done using with the CIGRE-601 standard and the IEEE-738 standard.
"""

###############################################################################
# Imports and utilities
# ^^^^^^^^^^^^^^^^^^^^^

import matplotlib.pyplot as plt
import numpy as np

from linerate.equations import cigre601, ieee738, solar_angles
from linerate.equations.math import switch_cos_sin

###############################################################################
# Simulation parameters
# ^^^^^^^^^^^^^^^^^^^^^
time = np.datetime64("2022-06-01T13:00")

omega = solar_angles.compute_hour_angle_relative_to_noon(time)
delta = solar_angles.compute_solar_declination(time)

vals_with_range = {
    "D": np.linspace(0.011, 0.038)[:, np.newaxis],
    "sin_H_s": np.linspace(-0.05, 1)[:, np.newaxis],
}

for k, v in vals_with_range.items():
    alpha_s = 0.8  # alpha in IEEE
    phi = 60  # Lat in IEEE
    gamma_c = 0  # Z_l i IEEE
    y = 0  # H_e in IEEE
    D = 0.025  # D_0 in IEEE
    F = 0.1
    N_s = 1
    sin_H_s = solar_angles.compute_sin_solar_altitude(phi, delta, omega)

    dict_in_loop = {k: v}
    locals().update(dict_in_loop)

    ###############################################################################
    # CIGRE601 calculations
    # ^^^^^^^^^^^^^^^^^^^^^^^^^

    chi = solar_angles.compute_solar_azimuth_variable(phi, delta, omega)
    C = solar_angles.compute_solar_azimuth_constant(chi, omega)
    Z_c = solar_angles.compute_solar_azimuth(C, chi)
    cos_eta = solar_angles.compute_cos_solar_effective_incidence_angle(sin_H_s, Z_c, gamma_c)
    sin_eta = switch_cos_sin(cos_eta)

    I_B = cigre601.solar_heating.compute_direct_solar_radiation(sin_H_s, N_s, y)
    I_d = cigre601.solar_heating.compute_diffuse_sky_radiation(I_B, sin_H_s)
    I_T = cigre601.solar_heating.compute_global_radiation_intensity(I_B, I_d, F, sin_eta, sin_H_s)

    ###############################################################################
    # IEEE738 calculations
    # ^^^^^^^^^^^^^^^^^^^^

    Q_s = ieee738.solar_heating.compute_total_heat_flux_density(sin_H_s, True)
    K_solar = ieee738.solar_heating.compute_solar_altitude_correction_factor(y)
    Q_se = ieee738.solar_heating.compute_elevation_correction_factor(K_solar, Q_s)
    chi = solar_angles.compute_solar_azimuth_variable(phi, delta, omega)
    C = solar_angles.compute_solar_azimuth_constant(chi, omega)
    Z_c = solar_angles.compute_solar_azimuth(C, chi)
    cos_theta = solar_angles.compute_cos_solar_effective_incidence_angle(sin_H_s, Z_c, gamma_c)

    ###############################################################################
    # Calculate P_c with different varying parameters
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    P_s_cigre = cigre601.solar_heating.compute_solar_heating(alpha_s, I_T, D)
    P_s_ieee = ieee738.solar_heating.compute_solar_heating(alpha_s, Q_se, cos_theta, D)

    ###############################################################################
    # Create visualisation
    # ^^^^^^^^^^^^^^^^^^^^

    fig = plt.figure()
    plt.plot(globals()[k][:, 0], P_s_cigre[:, 0], color="g", label="CIGRE")
    plt.plot(globals()[k][:, 0], P_s_ieee[:, 0], color="orange", label="IEEE")
    if k == "D":
        plt.xlabel("Conductor outer diameter [m]")
        plt.title(
            r"Solar heating calculated using the CIGRE-601 and the IEEE-736 standards."
            "\n"
            r"The only varying parameter in the calculations was conductor diameter.",
            fontsize=11,
        )
    elif k == "sin_H_s":
        plt.xlabel("Solar altitude [$^\\circ$]")
        plt.xticks(ticks=[0, 1 / 3, 2 / 3, 1], labels=[0, 30, 60, 90])
        plt.title(
            r"Solar heating calculated using the CIGRE-601 and the IEEE-736 standards."
            "\n"
            r"The only varying parameter in the calculations was solar altitude.",
            fontsize=11,
        )

    plt.legend()
    plt.xlim(np.min(globals()[k][:, 0]), np.max(globals()[k][:, 0]))
    plt.ylabel("Solar heating [W/m]")
    plt.ylim(-2, 32)
    plt.show()
