"""
Standard example from CIGRE TB 601
----------------------------------
In this example, we see how we can use ``linerate`` to compute the steady state thermal rating
and the steady state conductor temperature with the conductor, span and weather parameters listed
in Example B on page 79-81 of :cite:p:`cigre601`.
"""

###############################################################################
# Imports and utilities
# ^^^^^^^^^^^^^^^^^^^^^

import matplotlib.pyplot as plt
import numpy as np

import linerate

###############################################################################
# Define conductor type
# ^^^^^^^^^^^^^^^^^^^^^

conductor = linerate.Conductor(
    core_diameter=10.4e-3,
    conductor_diameter=28.1e-3,
    outer_layer_strand_diameter=2.2e-3,
    emissivity=0.9,
    solar_absorptivity=0.9,
    temperature1=25,
    temperature2=75,
    resistance_at_temperature1=7.283e-5,
    resistance_at_temperature2=8.688e-5,
    aluminium_cross_section_area=float("nan"),  # No core magnetisation loss
    constant_magnetic_effect=1,
    current_density_proportional_magnetic_effect=0,
    max_magnetic_core_relative_resistance_increase=1,
)


###############################################################################
# Create towers for a span that faces east-west
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

start_tower = linerate.Tower(latitude=50 - 0.0045, longitude=0, altitude=500 - 88)
end_tower = linerate.Tower(latitude=50 + 0.0045, longitude=0, altitude=500 + 88)
span = linerate.Span(
    conductor=conductor,
    start_tower=start_tower,
    end_tower=end_tower,
    ground_albedo=0.15,
    num_conductors=1,
)


###############################################################################
# Create the weather data class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

weather = linerate.Weather(
    air_temperature=20,
    wind_direction=np.radians(80),  # Conductor azimuth is 0, so angle of attack is 80
    wind_speed=1.66,
    clearness_ratio=0.5,
)


###############################################################################
# Setup CIGRE 601 model
# ^^^^^^^^^^^^^^^^^^^^^

time_of_measurement = np.datetime64("2016-10-03 14:00")
model = linerate.Cigre601(span, weather, time_of_measurement)

###############################################################################
# Compute line rating and temperature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

max_conductor_temperature = np.linspace(0, 100, 101)
current_load = np.linspace(0, 1_000, 101)

conductor_rating = model.compute_steady_state_ampacity(max_conductor_temperature)
conductor_temperature = model.compute_conductor_temperature(current_load, tolerance=0.01)

###############################################################################
# Plot line rating and temperature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), tight_layout=True)
axes[0].plot(max_conductor_temperature, conductor_rating, "k")
axes[0].set_xlabel(r"Max conductor temperature $[^\circ \mathrm{C}]$")
axes[0].set_ylabel(r"Ampacity rating $[\mathrm{A}]$")

axes[1].plot(current_load, conductor_temperature, "k")
axes[1].set_xlabel(r"Current $[\mathrm{A}]$")
axes[1].set_ylabel(r"Conductor temperature $[^\circ \mathrm{C}]$")

plt.show()
