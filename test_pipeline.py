import sys
sys.stdout.reconfigure(encoding="utf-8")
from src.data_preprocessing import load_data, clean_and_preprocess, get_dataset_summary
from src.ml_model import TextileMLSystem
from src.digital_twin import TextileFactoryDigitalTwin
from src.sustainability import calculate_sustainability_metrics
from src.recommendations import detect_bottlenecks, generate_recommendations
from src.gemini_assistant import query_gemini_process_intelligence
"""End-to-end checks for the textile intelligence pipeline."""

from src.data_preprocessing import clean_and_preprocess, get_dataset_summary, load_data
from src.digital_twin import TextileFactoryDigitalTwin
from src.ml_model import TextileMLSystem
from src.recommendations import detect_bottlenecks, generate_recommendations
from src.sustainability import calculate_sustainability_metrics


def test_end_to_end_pipeline():
    raw = load_data()
    clean = clean_and_preprocess(raw)
    summary = get_dataset_summary(clean)
    prep = summary["preprocessing_summary"]

    assert len(raw) == prep["original_rows"]
    assert len(clean) == prep["final_rows"]
    assert prep["aggregate_rows_removed"] > 0
    assert not any(clean[column].astype("string").str.upper().eq("TOTAL").any() for column in clean.select_dtypes(include=["object", "string"]).columns)

    ml = TextileMLSystem()
    clf_metrics, reg_metrics = ml.train_models(clean, dataset_name="weaving_dataset_full.csv")
    assert ml.is_trained
    assert clf_metrics["confusion_matrix"]
    assert {"mae", "rmse", "r2"}.issubset(reg_metrics)
    assert "Rec_Beam_length(yds)" not in ml.feature_names

    prediction = ml.predict_single({"epi": 110, "ppi": 80, "weft_count": 80, "warp_count_num": 40})
    assert "rejection_probability" in prediction
    assert "predicted_production_yds" in prediction

    reloaded = TextileMLSystem()
    assert reloaded.load_models("weaving_dataset_full.csv", ml.metadata["dataset_fingerprint"])
    assert "predicted_production_yds" in reloaded.predict_single({"epi": 110, "ppi": 80})

    baseline = TextileFactoryDigitalTwin(sim_duration_hours=12.0, random_seed=42).run()
    what_if = TextileFactoryDigitalTwin(
        sim_duration_hours=12.0, weaving_machines=8, defect_prob=0.04, random_seed=42
    ).run()
    baseline_sustainability = calculate_sustainability_metrics(baseline)
    what_if_sustainability = calculate_sustainability_metrics(what_if)
    assert baseline["stage_metrics"]
    assert what_if["completed_yds"] >= 0
    assert 0 <= baseline_sustainability["eco_score"] <= 100
    assert 0 <= what_if_sustainability["eco_score"] <= 100

    bottleneck = detect_bottlenecks(baseline)
    recommendations = generate_recommendations(
        baseline,
        baseline_sustainability,
        ml_context=prediction,
        feature_importances=ml.feature_importances_clf,
    )
    assert bottleneck and bottleneck["stage"] in baseline["stage_metrics"]
    assert recommendations
    assert all("supporting_metric" in recommendation for recommendation in recommendations)

    print("End-to-end pipeline checks passed.")
    print(f"Raw rows: {len(raw):,}; analytical rows: {len(clean):,}")
    print(f"Classification metrics: {clf_metrics}")
    print(f"Regression metrics: {reg_metrics}")
    print(f"Eco Score: {baseline_sustainability['eco_score']}/100")


if __name__ == "__main__":
    test_end_to_end_pipeline()
