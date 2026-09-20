"""
Bottleneck Detection & Recommendation Engine.
Analyzes Digital Twin simulation results and sustainability KPIs to produce explainable operational advice.
"""

def detect_bottlenecks(sim_results):
    """
    Identifies the process stage acting as the production bottleneck.
    Returns:
        dict: Bottleneck stage details, utilization %, avg wait time, and bottleneck severity.
    """
    stage_metrics = sim_results.get("stage_metrics", {})
    if not stage_metrics:
        return None

    # Find highest utilization stage
    max_util_stage = max(stage_metrics.items(), key=lambda x: x[1]["utilization_pct"])
    # Find highest waiting time stage
    max_wait_stage = max(stage_metrics.items(), key=lambda x: x[1]["avg_wait_hours"])

    primary_stage_name = max_util_stage[0]
    primary_util = max_util_stage[1]["utilization_pct"]
    primary_wait = max_util_stage[1]["avg_wait_hours"]
    machines = max_util_stage[1]["machines"]

    if primary_util >= 90.0:
        severity = "🔴 CRITICAL BOTTLENECK"
        summary = f"Severe congestion detected at the {primary_stage_name} stage ({primary_util}% utilization). Upstream buffer starvation likely."
    elif primary_util >= 78.0:
        severity = "🟡 MODERATE BOTTLENECK"
        summary = f"{primary_stage_name} is operating near peak capacity ({primary_util}% utilization, {primary_wait} hrs avg wait)."
    else:
        severity = "🟢 BALANCED FLOW"
        summary = f"All production stages maintain balanced throughput with no critical backlogs."

    return {
        "stage": primary_stage_name,
        "utilization_pct": primary_util,
        "avg_wait_hours": primary_wait,
        "machines": machines,
        "severity": severity,
        "summary": summary,
        "max_wait_stage": max_wait_stage[0],
        "max_wait_hours": max_wait_stage[1]["avg_wait_hours"],
    }

def generate_recommendations(sim_results, sustainability_results, ml_context=None, feature_importances=None):
    """
    Generates rule-based, actionable engineering recommendations based on calculated values.
    Returns:
        list of dicts: [ { "category": ..., "title": ..., "detail": ..., "impact": ... } ]
    """
    recs = []
    bottleneck = detect_bottlenecks(sim_results)
    stage_metrics = sim_results.get("stage_metrics", {})
    rejection_rate = sim_results.get("rejection_rate_pct", 0.0)
    eco_score = sustainability_results.get("eco_score", 75.0)
    energy_per_yd = sustainability_results.get("energy_per_yard_kwh", 0.5)
    waste_per_yd = sustainability_results.get("waste_per_yard_kg", 0.02)
    ml_context = ml_context or {}
    rejection_probability = ml_context.get("rejection_probability")
    predicted_production = ml_context.get("predicted_production_yds")
    top_ml_feature = next(iter(feature_importances or {}), None)

    # 1. Bottleneck & Capacity Recommendation
    if bottleneck and bottleneck["utilization_pct"] >= 78.0:
        stage = bottleneck["stage"]
        current_mc = bottleneck["machines"]
        recs.append(
            {
                "category": "🏭 Line Balancing & Throughput",
                "title": f"Expand Capacity at {stage} Stage",
                "detail": f"The {stage} stage has reached {bottleneck['utilization_pct']}% utilization with {bottleneck['avg_wait_hours']} hrs average queue delay. Adding 1 machine (from {current_mc} to {current_mc + 1}) or reducing cycle time by 15% will eliminate downstream starvation.",
                "priority": "High" if bottleneck["utilization_pct"] >= 88.0 else "Medium",
                "expected_gain": "Measure with the What-If scenario",
                "supporting_metric": f"{bottleneck['utilization_pct']}% utilization; {bottleneck['avg_wait_hours']} hr average wait",
            }
        )

    # 2. Quality & Defect Mitigation Recommendation
    if rejection_rate > 7.0:
        recs.append(
            {
                "category": "🧵 Quality & Defect Prevention",
                "title": "Optimize Sizing Starch Consistency & Warp Tension",
                "detail": f"Simulation indicates an elevated rejection rate of {rejection_rate}%. The model predicts {rejection_probability}% rejection probability for the current input" if rejection_probability is not None else f"Simulation indicates an elevated rejection rate of {rejection_rate}%. Implement automated yarn tension control and inspect sizing moisture content.",
                "priority": "High",
                "expected_gain": "Validate quality impact with a controlled scenario",
                "supporting_metric": f"Simulation rejection: {rejection_rate}%; model probability: {rejection_probability if rejection_probability is not None else 'N/A'}%",
            }
        )
    else:
        recs.append(
            {
                "category": "🧵 Quality & Defect Prevention",
                "title": "Maintain Predictive Quality Standard",
                "detail": f"Current rejection rate is well-controlled at {rejection_rate}%. Continue predictive screening on incoming yarn counts to monitor batch variance.",
                "priority": "Low",
                "expected_gain": "Monitor rejection stability",
                "supporting_metric": f"Simulation rejection: {rejection_rate}%; model probability: {rejection_probability if rejection_probability is not None else 'N/A'}%",
            }
        )

    # 3. Energy Efficiency Recommendation
    if energy_per_yd > 0.65:
        recs.append(
            {
                "category": "⚡ Energy Optimization",
                "title": "Install VFD Drives & Synchronize Idle Power",
                "detail": f"Specific energy consumption is {energy_per_yd} kWh/yard. Weaving and sizing motors draw standby power during buffer waits. Implement Variable Frequency Drives (VFDs) and automatic motor standby shutdown on idle looms.",
                "priority": "Medium",
                "expected_gain": "Validate energy impact with a controlled scenario",
                "supporting_metric": f"{energy_per_yd} kWh/yard estimated intensity",
            }
        )
    else:
        recs.append(
            {
                "category": "⚡ Energy Optimization",
                "title": "Energy Load Profile Healthy",
                "detail": f"Energy consumption of {energy_per_yd} kWh/yard is within optimal manufacturing parameters.",
                "priority": "Low",
                "expected_gain": "Maintain the current estimated intensity",
                "supporting_metric": f"{energy_per_yd} kWh/yard estimated intensity",
            }
        )

    # 4. Eco Score Enhancement
    if eco_score < 75.0:
        recs.append(
            {
                "category": "🌿 Sustainability & Circularity",
                "title": "Boost Project Eco Score with Fabric Scrap Recycling",
                "detail": f"Project Eco Score is currently {eco_score}/100. Route selvage trim and rejected yardage ({sustainability_results.get('total_waste_kg')} kg) to secondary fiber shredding for industrial batting to reclaim up to 8 points on circularity.",
                "priority": "Medium",
                "expected_gain": "Compare Eco Score in What-If simulation",
                "supporting_metric": f"Eco Score: {eco_score}/100; waste: {sustainability_results.get('total_waste_kg', 0)} kg",
            }
        )

    if predicted_production is not None:
        recs.append(
            {
                "category": "🤖 Model Decision Support",
                "title": "Use the production estimate as a planning signal",
                "detail": f"The regression model estimates {predicted_production:,.0f} yards for the entered parameters. This is a prediction, not a measured production result; compare it with the SimPy scenario output.",
                "priority": "Low",
                "expected_gain": "Improved planning visibility",
                "supporting_metric": f"Predicted production: {predicted_production:,.0f} yards" + (f"; top associated feature: {top_ml_feature}" if top_ml_feature else ""),
            }
        )

    return recs
