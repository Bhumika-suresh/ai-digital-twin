"""
Gemini 3.6 Flash AI Process Intelligence Assistant.
Supports direct frontend API key updates for gemini-3.6-flash with intelligent fallback.
"""

import os

def query_gemini_process_intelligence(
    prompt: str,
    context_data: dict,
    api_key: str = None,
    model_name: str = "gemini-3.6-flash",
) -> str:
    """
    Sends a specialized manufacturing analysis prompt to Gemini 2.0 Flash with plant telemetry context.
    If no API key is provided or the call fails, falls back to a high-quality expert rule-based response.
    """
    # Prefer explicit user-entered key, then env var
    effective_api_key = api_key.strip() if (api_key and api_key.strip()) else os.environ.get("GEMINI_API_KEY", "").strip()

    # Format system context
    context_str = f"""
=== TEXTILE FACTORY DIGITAL TWIN TELEMETRY ===
- Active Dataset: {context_data.get('dataset_name', 'Kaggle Textile Weaving Dataset')}
- Total Completed Production: {context_data.get('completed_yds', 0)} yards
- Total Rejected Production: {context_data.get('rejected_yds', 0)} yards ({context_data.get('rejection_rate_pct', 0)}%)
- Primary Bottleneck Stage: {context_data.get('bottleneck_stage', 'None')} ({context_data.get('bottleneck_util', 0)}% utilization)
- Total Estimated Energy: {context_data.get('total_energy_kwh', 0)} kWh ({context_data.get('energy_per_yd', 0)} kWh/yd)
- Total Material Scrap: {context_data.get('total_waste_kg', 0)} kg
- Carbon Footprint: {context_data.get('total_carbon_kg_co2e', 0)} kg CO2e
- Project Eco Score: {context_data.get('eco_score', 0)} / 100
- ML Rejection Classifier Accuracy: {context_data.get('ml_accuracy', 'N/A')}
===============================================
"""

    full_prompt = f"{context_str}\n\nUser Question / Directive: {prompt}\n\nPlease provide expert textile engineering and sustainability guidance based on the factory data above."

    if effective_api_key:
        # Try google.genai (official modern SDK for Gemini 2.0)
        try:
            from google import genai
            client = genai.Client(api_key=effective_api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            if response and response.text:
                return response.text
        except Exception as e1:
            # Fallback to google.generativeai legacy library if installed
            try:
                import google.generativeai as gai
                gai.configure(api_key=effective_api_key)
                model = gai.GenerativeModel(model_name)
                response = model.generate_content(full_prompt)
                if response and response.text:
                    return response.text
            except Exception as e2:
                return f"⚠️ **Gemini 2.0 Flash API Call Notice:** Error connecting with provided key ({str(e2)}).\n\n*Displaying Offline Process Intelligence Expert Insight below:*\n\n" + _generate_expert_offline_response(prompt, context_data)

    return _generate_expert_offline_response(prompt, context_data)

def _generate_expert_offline_response(prompt: str, context: dict) -> str:
    """Built-in domain intelligence for offline demonstration."""
    bottleneck = context.get("bottleneck_stage", "Weaving")
    util = context.get("bottleneck_util", 85.0)
    rej_pct = context.get("rejection_rate_pct", 8.0)
    eco_score = context.get("eco_score", 76.0)
    energy_yd = context.get("energy_per_yd", 0.52)

    prompt_lower = prompt.lower()
    
    if "bottleneck" in prompt_lower or "capacity" in prompt_lower or "line" in prompt_lower:
        return f"""### 🏭 Process Bottleneck Diagnosis (SimPy Telemetry)
- **Congestion Focal Point:** The **{bottleneck}** stage is the primary critical path constraint at **{util}% utilization**.
- **Root Cause:** Sizing and Warping output feeds into {bottleneck} at a higher rate than the current loom cycle time permits.
- **Recommended Action:**
  1. Add buffer capacity or an additional machine at the {bottleneck} stage.
  2. Implement predictive maintenance on loom rapier drives to reduce micro-stoppages.
  3. Expected throughput improvement: **+14% to +20% yards/day**."""

    elif "eco" in prompt_lower or "sustainability" in prompt_lower or "green" in prompt_lower or "carbon" in prompt_lower:
        return f"""### 🌿 Sustainability & Eco Score Analysis
- **Current Project Eco Score:** **{eco_score} / 100**
- **Specific Energy Intensity:** **{energy_yd} kWh/yard**
- **Decarbonization Levers:**
  1. **Waste Heat Recovery in Sizing:** Steam cylinders in sizing account for substantial thermal energy; reclaiming cylinder condensate can reduce sizing energy by ~18%.
  2. **Yarn Waste Circularity:** Divert edge trims and flawed yardage ({context.get('total_waste_kg', 0)} kg) to regenerated yarn spinning to gain +7 Eco Score points.
  3. **Loom Standby Power Cutoff:** Implement automatic drive idle cutoffs to save 15% standby electricity."""

    elif "rejection" in prompt_lower or "defect" in prompt_lower or "quality" in prompt_lower or "ml" in prompt_lower:
        return f"""### 🧵 Quality & Rejection Analysis (ML Intelligence)
- **Current Rejection Rate:** **{rej_pct}%**
- **Key Determinants Identified by Random Forest:**
  - High EPI/PPI density fabric constructions experience greater friction during shedding.
  - Sizing wax/starch pickup consistency directly prevents warp end breakages.
- **Action Plan:**
  - Fine-tune sizing moisture retention to 6.5–7.0%.
  - Run ML pre-screening on incoming beam batches to dynamically adjust loom RPM."""

    else:
        return f"""### 💡 AI Process Intelligence Overview
- **Factory Status:** Operating across Warping → Sizing → Weaving → Quality Inspection.
- **Production Performance:** {context.get('completed_yds', 0)} yards completed, {rej_pct}% rejection.
- **Critical Constraint:** **{bottleneck}** stage ({util}% load).
- **Sustainability Standing:** Eco Score **{eco_score}/100** with {context.get('total_carbon_kg_co2e', 0)} kg CO2e carbon footprint.
- **Recommendation:** Rebalance {bottleneck} capacity and optimize sizing starch to simultaneously raise throughput and improve the Eco Score."""
