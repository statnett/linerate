from linerate.units import (
    Celsius,
    JoulePerKilogramPerKelvin,
    KilogramPerMeter,
    PerKelvin,
    JoulePerMeter,
)


def calculate_heat_capacity_per_unit_length(
    mass_per_unit_length: KilogramPerMeter,
    specific_heat_capacity_at_20_celsius: JoulePerKilogramPerKelvin,
    specific_heat_capacity_coefficient: PerKelvin,
    conductor_temperature: Celsius,
) -> JoulePerMeter:
    m = mass_per_unit_length
    c = specific_heat_capacity_at_20_celsius
    T = conductor_temperature
    beta = specific_heat_capacity_coefficient
    return m * c * (1 + (T - 20) * beta)
