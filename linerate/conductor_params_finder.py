import os
from typing import Protocol
import pandas as pd

from linerate.domain import ConductorParameters


class ConductorFinderProtocol(Protocol):
    def find_conductor_parameters_by_names(self, name_series: pd.Series) -> pd.DataFrame:
        pass


class ConductorFinder:
    def __init__(self):
        all_params = self.load_from_csv()
        self.params_map = {param.code: param for param in all_params}

        pass

    def find_conductor_parameters_by_names(self, name_series: pd.Series) -> pd.DataFrame:
        if not isinstance(name_series, pd.Series):
            raise ValueError("Input must be a pandas Series")

        parameters_list = []
        for name in name_series:
            if name in self.params_map:
                parameters_list.append(dict(self.params_map[name]))
            else:
                raise ValueError(f"Conductor {name} not found")

        return pd.DataFrame(parameters_list)

    @staticmethod
    def load_from_csv():
        # Format in CSV:
        # code,old_code,area_al_sq_mm,area_steel_sq_mm,area_total_sq_mm,no_of_wires_al,no_of_wires_steel,wire_diameter_al_mm,wire_diameter_steel_mm,core_diameter_mm,conductor_diameter_mm,mass_per_unit_length_kg_per_km,rated_strength_kn,dc_resistance_low_temperature_ohm_per_km,final_modulus_of_elasticity_n_per_sq_mm,coefficient_of_linear_expansion_per_k,current_carrying_capacity_a,country,aluminium_type,low_temperature_deg_c,high_temperature_deg_c,dc_resistance_high_temperature_ohm_per_km
        # 152-AL1/25-ST1A,ACSR152/25 OSTRICH,152.2,24.7,176.9,26,7,2.73,2.12,6.36,17.28,613.6,54.78,0.1898,81000,1.92e-05,0,Finland,AL1,20,80,0.235352
        #
        csv_file_path = os.path.join(os.path.dirname(__file__), 'conductors_standard.csv')
        df = pd.read_csv(csv_file_path)
        return ConductorParameters.from_dataframe(df)
