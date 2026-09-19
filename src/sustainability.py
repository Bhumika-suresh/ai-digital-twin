"""
Sustainability & Project Eco Score estimation module for Textile Manufacturing.
Calculates transparent estimated metrics for Energy, Water, Material Waste, Carbon, and composite Project Eco Score.
"""

import numpy as np

# Default Engineering Constants & Machine Power Ratings (kW)
DEFAULT_STAGE_POWER_KW = {
    "Warping": 7.5,      # Creel & beam winder motors
    "Sizing": 18.0,      # Sizing box, steam drying cylinder motors
    "Weaving": 5.5,      # High-speed air-jet / rapier loom
    "Inspection": 2.0,   # Inspection table, illumination & fabric drive
}

DEFAULT_GRID_EMISSION_FACTOR = 0.52   # kg CO2e per kWh (Standard regional industrial grid)
DEFAULT_WATER_COEFF_L_PER_YD = 1.2    # Liters of water per yard (sizing preparation & yarn conditioning)
DEFAULT_FABRIC_WEIGHT_KG_PER_YD = 0.18 # Standard medium fabric weight (~180 GSM at 44" width)
DEFAULT_SELVAGE_TRIM_PCT = 2.0        # Expected 2% fabric edge trim allowance

def calculate_sustainability_metrics(
    sim_results,
    stage_power_kw=None,
    grid_emission_factor=DEFAULT_GRID_EMISSION_FACTOR,
    water_coeff_l_per_yd=DEFAULT_WATER_COEFF_L_PER_YD,
    fabric_weight_kg_per_yd=DEFAULT_FABRIC_WEIGHT_KG_PER_YD,
    selvage_trim_pct=DEFAULT_SELVAGE_TRIM_PCT,
):
    """
    Computes sustainability indicators based on simulation duration and machine utilization.
    Returns:
        dict: Detailed breakdown of estimated Energy, Water, Waste, Carbon, and Eco Score.
    """
    if stage_power_kw is None:
        stage_power_kw = DEFAULT_STAGE_POWER_KW

    sim_duration = sim_results.get("sim_duration_hours", 48.0)
    total_produced_yds = sim_results.get("total_produced_yds", 0.0)
    completed_yds = sim_results.get("completed_yds", 0.0)
    rejected_yds = sim_results.get("rejected_yds", 0.0)
    rejection_rate_pct = sim_results.get("rejection_rate_pct", 0.0)
    stage_metrics = sim_results.get("stage_metrics", {})

    # 1. Energy Calculation (kWh) = Machine Count * Utilization * Power (kW) * Duration (Hrs)
    energy_breakdown_kwh = {}
    total_energy_kwh = 0.0

    for stage, power_kw in stage_power_kw.items():
        stage_info = stage_metrics.get(stage, {})
        machines = stage_info.get("machines", 1)
        utilization_ratio = stage_info.get("utilization_pct", 50.0) / 100.0
        
        # Machine operates during busy time, with 15% standby power when idle
        active_hours = sim_duration * utilization_ratio * machines
        standby_hours = sim_duration * (1.0 - utilization_ratio) * machines
        
        stage_kwh = (active_hours * power_kw) + (standby_hours * power_kw * 0.15)
        energy_breakdown_kwh[stage] = round(stage_kwh, 2)
        total_energy_kwh += stage_kwh

    total_energy_kwh = round(total_energy_kwh, 2)
    energy_per_yard_kwh = round(total_energy_kwh / max(1.0, completed_yds), 3)

    # 2. Water Calculation (Liters)
    total_water_liters = round(total_produced_yds * water_coeff_l_per_yd, 1)
    water_per_yard_l = round(total_water_liters / max(1.0, completed_yds), 2)

    # 3. Material Waste Calculation (kg)
    # Waste from rejected yardage + selvage trim scrap
    rejected_waste_kg = rejected_yds * fabric_weight_kg_per_yd
    trim_waste_kg = total_produced_yds * (selvage_trim_pct / 100.0) * fabric_weight_kg_per_yd
    total_waste_kg = round(rejected_waste_kg + trim_waste_kg, 2)
    waste_per_yard_kg = round(total_waste_kg / max(1.0, completed_yds), 4)

    # 4. Carbon Footprint (kg CO2e)
    # Electricity emissions + minimal water processing carbon (0.0003 kg CO2e / L)
    carbon_from_energy = total_energy_kwh * grid_emission_factor
    carbon_from_water = total_water_liters * 0.0003
    total_carbon_kg_co2e = round(carbon_from_energy + carbon_from_water, 2)
    carbon_per_yard_kg_co2e = round(total_carbon_kg_co2e / max(1.0, completed_yds), 3)

    # 5. Project Eco Score (0 - 100)
    # 4 Balanced Pillars (25 pts each):
    # Pillar A: Energy Efficiency (Target: < 0.45 kWh/yd -> 25 pts, > 1.2 kWh/yd -> 5 pts)
    energy_score = max(5.0, min(25.0, 25.0 - ((energy_per_yard_kwh - 0.40) * 20.0)))

    # Pillar B: Waste & Rejection Quality (Target: < 3% rejection -> 25 pts, > 20% -> 5 pts)
    waste_score = max(5.0, min(25.0, 25.0 - (rejection_rate_pct * 1.0)))

    # Pillar C: Carbon Intensity (Target: < 0.25 kg CO2e/yd -> 25 pts, > 0.70 -> 5 pts)
    carbon_score = max(5.0, min(25.0, 25.0 - ((carbon_per_yard_kg_co2e - 0.20) * 35.0)))

    # Pillar D: Machine Throughput & Process Health (Balanced avg utilization between 65% and 85%)
    avg_util = np.mean([s.get("utilization_pct", 50.0) for s in stage_metrics.values()]) if stage_metrics else 70.0
    if 65.0 <= avg_util <= 88.0:
        util_score = 25.0
    elif avg_util < 65.0:
        util_score = max(8.0, 25.0 - (65.0 - avg_util) * 0.4)
    else:
        util_score = max(10.0, 25.0 - (avg_util - 88.0) * 0.8)

    eco_score = round(energy_score + waste_score + carbon_score + util_score, 1)
    eco_score = max(0.0, min(100.0, eco_score))

    return {
        "total_energy_kwh": total_energy_kwh,
        "energy_breakdown_kwh": energy_breakdown_kwh,
        "energy_per_yard_kwh": energy_per_yard_kwh,
        "total_water_liters": total_water_liters,
        "water_per_yard_l": water_per_yard_l,
        "total_waste_kg": total_waste_kg,
        "rejected_waste_kg": round(rejected_waste_kg, 2),
        "trim_waste_kg": round(trim_waste_kg, 2),
        "waste_per_yard_kg": waste_per_yard_kg,
        "total_carbon_kg_co2e": total_carbon_kg_co2e,
        "carbon_per_yard_kg_co2e": carbon_per_yard_kg_co2e,
        "eco_score": eco_score,
        "eco_score_breakdown": {
            "energy_efficiency": round(energy_score, 1),
            "waste_minimization": round(waste_score, 1),
            "carbon_intensity": round(carbon_score, 1),
            "throughput_health": round(util_score, 1),
        },
        "assumptions": {
            "grid_emission_factor": f"{grid_emission_factor} kg CO2e / kWh",
            "water_consumption_rate": f"{water_coeff_l_per_yd} L / yard",
            "fabric_areal_weight": f"{fabric_weight_kg_per_yd} kg / yard",
            "selvage_trim_allowance": f"{selvage_trim_pct}%",
            "machine_powers_kw": stage_power_kw,
        },
    }
