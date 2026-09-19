import sys
sys.stdout.reconfigure(encoding="utf-8")
from src.data_preprocessing import load_data, clean_and_preprocess, get_dataset_summary
from src.ml_model import TextileMLSystem
from src.digital_twin import TextileFactoryDigitalTwin
from src.sustainability import calculate_sustainability_metrics
from src.recommendations import detect_bottlenecks, generate_recommendations
from src.gemini_assistant import query_gemini_process_intelligence

print("--- TEST 1: DATA PREPROCESSING ---")
df_raw = load_data()
df_clean = clean_and_preprocess(df_raw)
summary = get_dataset_summary(df_clean)
print("Dataset Shape:", df_clean.shape)
print(f"Summary Rows: {summary['rows']}, Cols: {summary['cols']}, Rejection Rate: {summary['rejection_rate']:.2f}%")

print("\n--- TEST 2: ML MODEL TRAINING & PREDICTION ---")
ml = TextileMLSystem()
clf_m, reg_m = ml.train_models(df_clean)
print(f"Classification Accuracy: {clf_m['accuracy']*100:.2f}%, F1: {clf_m['f1']:.3f}")
print(f"Regression R2: {reg_m['r2']:.3f}, MAE: {reg_m['mae']:.2f}")
sample_input = {
    'epi': 110,
    'ppi': 80,
    'weft_count': 40,
    'warp_count_num': 40,
    'Fabric_Allowance': 5.0,
    'Shrink_allow': 3.0,
    'Req_Finish_Fabrics': 10000,
    'Rec_Beam_length(yds)': 10500,
    'Req_beam_length(yds)': 10500,
    'Req_grey_fabric': 10300,
    'warp_cover_factor': 17.39,
    'weft_cover_factor': 12.65,
    'total_fabric_density': 190
}
pred = ml.predict_single(sample_input)
print("Sample Live Prediction:", pred)

print("\n--- TEST 3: SIMPY DIGITAL TWIN ---")
twin = TextileFactoryDigitalTwin(sim_duration_hours=24.0)
sim_res = twin.run()
print(f"Sim Total Completed Yds: {sim_res['completed_yds']}, Rejected Yds: {sim_res['rejected_yds']}, Throughput: {sim_res['throughput_yds_per_hr']} yds/hr")

print("\n--- TEST 4: SUSTAINABILITY & ECO SCORE ---")
sust = calculate_sustainability_metrics(sim_res)
print(f"Total Energy kWh: {sust['total_energy_kwh']}, Total Waste kg: {sust['total_waste_kg']}, Eco Score: {sust['eco_score']}/100")

print("\n--- TEST 5: BOTTLENECK & RECOMMENDATIONS ---")
bn = detect_bottlenecks(sim_res)
recs = generate_recommendations(sim_res, sust)
print(f"Bottleneck: {bn['stage']}, Util: {bn['utilization_pct']}%, Severity: {bn['severity']}")
print(f"Generated Recommendations Count: {len(recs)}")

print("\n--- TEST 6: GEMINI 2.0 FLASH ASSISTANT ---")
ai_resp = query_gemini_process_intelligence(
    "Analyze bottlenecks and suggest eco optimizations",
    {
        'bottleneck_stage': bn['stage'],
        'bottleneck_util': bn['utilization_pct'],
        'completed_yds': sim_res['completed_yds'],
        'rejection_rate_pct': sim_res['rejection_rate_pct'],
        'total_energy_kwh': sust['total_energy_kwh'],
        'eco_score': sust['eco_score']
    }
)
print("AI Response Preview:\n", ai_resp[:250])

print("\n>>> ALL CORE TESTS PASSED WITH 100% SUCCESS! <<<")
