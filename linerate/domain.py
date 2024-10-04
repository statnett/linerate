from pydantic import BaseModel
import pandas as pd

class ConductorParameters(BaseModel):
    code: str
    old_code: str
    area_al_sq_mm: float            # mm² (area of aluminum strands)
    area_steel_sq_mm: float         # mm² (area of steel strands)
    area_total_sq_mm: float         # mm² (total cross-sectional area, including aluminum and steel)
    no_of_wires_al: float           # count (total number of aluminum wires)
    no_of_wires_steel: float        # count (total number of steel wires)
    wire_diameter_al_mm: float      # mm (diameter of aluminum strands)
    wire_diameter_steel_mm: float   # mm (diameter of steel strands)
    core_diameter_mm: float         # mm (diameter of the steel core)
    conductor_diameter_mm: float    # mm (actual conductor diameter)
    mass_per_unit_length_kg_per_km: float # kg/m
    rated_strength_kn: float        # kN (total tensile strength)
    dc_resistance_low_temperature_ohm_per_km: float # ohm/km (at lower temperature)
    final_modulus_of_elasticity_n_per_sq_mm: float  # N/mm² (final elastic modulus for steel and aluminum composite)
    coefficient_of_linear_expansion_per_k: float    # 1/°C
    current_carrying_capacity_a: float              # A
    country: str                                    # country code (ISO)
    aluminium_type: str                             # type of aluminum
    low_temperature_deg_c: float                    # °C (low operating temperature)
    high_temperature_deg_c: float                   # °C (high operating temperature)
    dc_resistance_high_temperature_ohm_per_km: float    # ohm/km (at higher temperature)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return [cls(**row) for row in df.to_dict(orient='records')]
