# Overview

A package containing functionality to compute ampacity line ratings for overhead lines.
Currently, the package only contains equations from CIGRE TB 601.

## Installation


```raw
pip install linerate
```

## Documentation

### Line rate calculation
Main usage is in the `linerate.helper` module. Use the compute_line_rating for calculating the line rating from pandas DataFrame.
This uses default conductor values from standard (values in linerate/conductor_standard.csv).

In order to use custom conductor values, use the LineRatingComputation class and pass in the conductor finder. For examples look at `tests/df_helper/test_helper.py`

```python
import numpy as np
import pandas as pd
from linerate.helper import compute_line_rating

df = pd.DataFrame({
    'line_id': [1],
    'span_number': ['45.6789_50.1234'],
    'timestamp': [np.datetime64('2024-01-01T00:00:00')],
    'temperature': [20.0],
    'wind_speed': [5.0],
    'wind_direction': [np.radians(90)],
    'humidity': [50.0],
    'solar_radiation_clouds': [800.0],
    'start_lon': [24.9384],
    'start_lat': [60.1699],
    'end_lon': [24.9484],
    'end_lat': [60.1799],
    'mid_lon': [24.9434],
    'mid_lat': [60.1749],
    'bearing': [45.0],
    'wires': [1],
    'max_allowed_temp': [60.0],
    'conductor': ['565-AL1/72-ST1A']
})
result = compute_line_rating(df)
assert isinstance(result, pd.Series)
```

### Solar irradiance clear sky calculation
The solar irradiance clear sky calculation is done in the `linerate.helper` module. The function `calculate_solar_irradiance(latitude, longitude, timestamps)` calculates the solar irradiance clear sky based on the latitude, longitude, and timestamp.

```python
import pandas as pd
from linerate.helper import calculate_solar_irradiance

datetime_idx = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 23:00:00', freq='h', tz='UTC')
df = pd.DataFrame({'timestamp': datetime_idx})
df['latitude'] = 60.1699
df['longitude'] = 24.9384

# Using this function, the dataframe has to be indexed according to timestamp
df.set_index('timestamp', inplace=True)
result = calculate_solar_irradiance(df['latitude'], df['longitude'], df.index)
df['irradiance'] = result
df.reset_index(inplace=True)
```

## Development

Dependencies for the project are managed with uv (https://docs.astral.sh/uv/getting-started/installation/)

```raw
uv sync
```

### Generate docs
??? Not sure if we need this.

### Release new version
Currently, we keep our version in git repository.
Change version in "pyproject.toml", commit and push to git.
```raw
git tag -v v2.1.0
git push --tags
```
