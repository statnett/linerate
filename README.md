# Overview

A package containing functionality to compute ampacity line ratings for overhead lines.
Currently, the package only contains equations from CIGRE TB 601.

## Installation

```
pip install linerate
```

## Documentation

This library is split into four main parts:

 1. The `equations` module, which contains one pure function for each equation in CIGRE TB 601,
 2. the `types` module, which contains datatypes for conductors, weather parameters and spans,
 3. the `model` module, which contains a wrapper class `Cigre601` to easily compute the ampacity and conductor temperature based on a `Span` and `Weather` instance,
 4. and the `solver` module, which contains a vectorized bisection solver for estimating the steady state ampacity and temperature of a conductor.

A typical user of this software package will only use the `types` and `model` module,
and the `model` module will then use functions from `equations` and `solver` to estimate the conductor temperature and ampacity. However, to understand the parameters, it may be useful to look at the functions
in the `equations` module, as we have taken care to ensure that the argument names stay consistent. 

Below, we see an example of how to compute the conductor temperature based on *Example B* on page 79-81 in CIGRE TB 601. 

```python
import numpy as np
import linerate


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


start_tower = linerate.Tower(latitude=50 - 0.0045, longitude=0, altitude=500 - 88)
end_tower = linerate.Tower(latitude=50 + 0.0045, longitude=0, altitude=500 + 88)
span = linerate.Span(
    conductor=conductor,
    start_tower=start_tower,
    end_tower=end_tower,
    ground_albedo=0.15,
    num_conductors=1,
)


weather = linerate.Weather(
    air_temperature=20,
    wind_direction=np.radians(80),  # Conductor azimuth is 0, so angle of attack is 80
    wind_speed=1.66,
    clearness_ratio=0.5,
)


time_of_measurement = np.datetime64("2016-10-03 14:00")
max_conductor_temperature = 100
current_load = 1000

model = linerate.Cigre601(span, weather, time_of_measurement)
conductor_rating = model.compute_steady_state_ampacity(max_conductor_temperature)
print(f"The span has a steady-state ampacity rating of {conductor_rating:.0f} A if the maximum temperature is {max_conductor_temperature} °C")
conductor_temperature = model.compute_conductor_temperature(current_load)
print(f"The conductor has a temperature of {conductor_temperature:.0f} °C when operated at {current_load} A")
```

## Transient state solver
There is currently no transient solver or short time thermal rating solver, but that is on the roadmap.

## Development

Dependencies for the project are managed with poetry.
To install all dependencies run:

```
poetry install
```

Remember that when developing a library it is *not* recommended to
commit the `poetry.lock` file.
