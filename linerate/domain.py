from pydantic import BaseModel

class ConductorParameters(BaseModel):
    name: str
    aluminium_area: float  # mm² (area of aluminum strands)
    total_area: float  # mm² (total cross-sectional area, including aluminum and steel)
    weight_per_unit_length: float  # N/m (weight of the conductor per meter)
    rated_tensile_strength: float  # kN (total tensile strength)
    horizontal_component_of_tension: float  # kN (20% of RTS)
    conductor_temperature_1: float  # °C (ambient or reference temperature)
    conductor_temperature_2: float  # °C (high operating temperature)
    final_modulus_of_elasticity_of_conductor: float  # N/mm² (final elastic modulus for steel and aluminum composite)
    initial_modulus_of_elasticity_of_conductor: float  # N/mm² (initial elastic modulus)
    nominal_diameter: float  # mm (overall conductor nominal diameter)
    conductor_diameter: float  # mm (actual conductor diameter, very close to nominal)
    number_of_wires: int  # count (total number of wires)
    outer_layer_strand_diameter: float  # mm (diameter of outer aluminum strands)
    steel_core_diameter: float  # mm (diameter of the steel core)
    sectional_area_of_steel_core: float  # mm² (cross-sectional area of the steel core)
    linear_mass: float  # kg/m (mass per meter)
    rated_strength: float  # daN (converted from kN; 174.14 kN = 17414 daN)
    coefficient_of_linear_expansion: float  # 1/°C (typical expansion coefficient for composite conductors)
    resistance_per_km: float  # ohm/km (at 20°C, corrected value)
    resistance_per_km_at_high_temperature: float  # ohm/km (at 80°C)
    solar_absorptivity: float  # dimensionless (solar radiation absorption factor)
    emissivity: float  # dimensionless (thermal emissivity of the conductor)
    specific_heat_capacity_of_steel_at_20c: float  # J/kg*K (specific heat capacity of steel)
    specific_heat_capacity_of_aluminum_at_20c: float  # J/kg*K (specific heat capacity of aluminum)
    temperature_coefficient_of_steel_specific_heat_capacity: float  # 1/K (temperature coefficient of steel specific heat capacity)
    temperature_coefficient_of_aluminum_specific_heat_capacity: float  # 1/K (temperature coefficient of aluminum specific heat capacity)
    weight_per_unit_length_aluminium: float  # kg/m (weight per meter of aluminum)
    weight_per_unit_length_steel: float  # kg/m (weight per meter of steel)
    constant_magnetic_effect: float  # 1
    current_density_proportional_magnetic_effect: float  # 0
    max_magnetic_core_relative_resistance_increase: float  # 1