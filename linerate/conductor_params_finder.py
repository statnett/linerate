import pandas as pd

from linerate.conductor_params_defaults import conductor_565


class ConductorFinder:
    def __init__(self):
        pass

    @staticmethod
    def find_conductor_parameters_by_names(name_series: pd.Series) -> pd.DataFrame:
        if not isinstance(name_series, pd.Series):
            raise ValueError("Input must be a pandas Series")

        parameters_list = []
        for name in name_series:
            if name == "565-AL1/72-ST1A":
                parameters_list.append(conductor_565.__dict__)
            else:
                raise ValueError(f"Conductor {name} not found")

        return pd.DataFrame(parameters_list)