import pandas as pd

from linerate.conductor_params_finder import ConductorFinder
import pytest

def test_conductor_params_dataframe():
    cf = ConductorFinder()
    input_series = pd.Series(['152-AL1/25-ST1A', '565-AL1/72-ST1A'])
    output_df = cf.find_conductor_parameters_by_names(input_series)
    assert output_df.loc[0]['code'] == '152-AL1/25-ST1A'
    assert output_df.loc[0]['area_al_sq_mm'] == 152.2
    assert output_df.loc[1]['code'] == '565-AL1/72-ST1A'
    assert output_df.loc[1]['area_al_sq_mm'] == 565.0
    print(output_df.loc[1])
