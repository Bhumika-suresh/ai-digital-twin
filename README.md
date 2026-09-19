# 🧵 AI-Powered Digital Twin for Sustainable Textile Manufacturing

A comprehensive Streamlit dashboard integrating real textile weaving data, Machine Learning, SimPy discrete-event simulation, sustainability analysis, and Gemini 3.6 Flash AI copilot.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Problem Statement

Textile manufacturing involves complex multi-stage processes (Warping → Sizing → Weaving → Inspection) where inefficiencies lead to high rejection rates, energy waste, and carbon emissions. This project creates an **AI-powered Digital Twin** to simulate, predict, and optimize textile production processes.

## 🏗️ Architecture

```
Raw Material → Warping → Sizing → Weaving → Quality Inspection → Finished Product
```

| Module | Technology | Purpose |
|--------|-----------|---------|
| Data Analysis | Pandas, Plotly | EDA, correlation, distribution analysis |
| ML Prediction | Scikit-learn RandomForest | Rejection classification + production regression |
| Digital Twin | SimPy | Discrete-event factory simulation |
| Sustainability | Custom formulas | Energy, water, waste, carbon estimation |
| Eco Score | Weighted composite | 4-pillar sustainability score (0-100) |
| AI Copilot | Gemini 3.6 Flash | Process intelligence & recommendations |

## 📊 Dataset

[Textile Weaving Dataset](https://data.mendeley.com/datasets/6mwgj7tms3/2) from Mendeley Data (auto-downloaded on first run):

- `weaving_rejection_dataset.csv` — 22,010 records, 14 columns (ML training)
- `weaving_dataset_full.csv` — 121,148 records, 19 columns (detailed analysis)

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/textile-digital-twin.git
cd textile-digital-twin

# Install dependencies
pip install -r requirements.txt

# Run the app (datasets auto-download on first run)
streamlit run app.py
```

### Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Click **Deploy** — done!

## 📁 Project Structure

```
textile-digital-twin/
├── app.py                    # Main Streamlit dashboard
├── setup_data.py             # Auto-download datasets
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── .gitignore
├── data/                     # Auto-populated with CSV datasets
├── models/                   # ML model artifacts (auto-generated)
└── src/
    ├── __init__.py
    ├── data_preprocessing.py # Data loading, cleaning, feature engineering
    ├── ml_model.py           # RandomForest Classifier + Regressor
    ├── digital_twin.py       # SimPy discrete-event simulation
    ├── sustainability.py     # Energy, Water, Waste, Carbon, Eco Score
    ├── recommendations.py    # Bottleneck detection + rule-based advice
    └── gemini_assistant.py   # Gemini 3.6 Flash AI with offline fallback
```

## 🎨 Dashboard Tabs

1. **📊 Dataset Overview** — EDA, statistics, EPI/PPI scatter, correlation heatmap
2. **🤖 ML Prediction** — Confusion matrix, feature importance, live predictor
3. **🏭 Digital Twin** — Machine utilization, queue delays, event log
4. **🔄 What-If Simulation** — Side-by-side scenario comparison
5. **🌿 Sustainability** — Energy, water, waste, carbon gauge + Eco Score
6. **⚡ Bottlenecks** — Auto-detected with actionable recommendations
7. **💬 AI Copilot** — Gemini 3.6 Flash with live factory telemetry

## 🔑 Gemini API Key

The AI Copilot tab uses Google Gemini 3.6 Flash. Enter your API key in the sidebar. Works without a key using offline expert mode.

## 📈 ML Performance

| Model | Metric | Score |
|-------|--------|-------|
| RandomForestClassifier | Accuracy | 87.1% |
| RandomForestClassifier | F1-Score | 90.9% |
| RandomForestRegressor | R² Score | 0.985 |

## 🧰 Technologies

- **Streamlit** — Interactive dashboard
- **SimPy** — Discrete-event simulation
- **Scikit-learn** — Machine learning
- **Plotly** — Interactive visualizations
- **Pandas / NumPy** — Data processing
- **Gemini 3.6 Flash** — AI process intelligence

## 📄 License

MIT License — Free for educational and research use.
