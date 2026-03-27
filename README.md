# ASTRAS - SAR Narrative Generator with Behavioral Intelligence

Behavioral Intelligence-Driven AML Detection System with XGBoost, SHAP Explainability, NetworkX Graph Analysis, and RAG-powered SAR Generation.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python run_pipeline.py
```
This runs all 8 phases (~12-15 min):
1. Synthetic data generation (1,000 customers, 10 AML typologies)
2. Behavioral signal computation (19 signals per customer)
3. NetworkX graph analysis (15 graph features)
4. BSI (Behavioral Stability Index) scoring
5. Adaptive monitoring classification
6. XGBoost meta-risk classifier training + SHAP
7. RAG template setup
8. Audit trail initialization

### 3. Launch Dashboard
```bash
streamlit run app.py
```

### 4. (Optional) Launch API
```bash
uvicorn api:app --reload
```

## Architecture

```
config.py                  # Central configuration
run_pipeline.py            # 8-phase pipeline orchestrator
app.py                     # Streamlit dashboard (5 pages)
api.py                     # FastAPI REST endpoints

services/

  data_generation/
    data_generator.py        # Synthetic AML data with gray-area customers

  behavioral_engine/
    bsi.py                   # Behavioral Stability Index (0-100)
    bsi_timeline.py          # Rolling-window BSI time series
    adaptive_monitor.py      # Dynamic monitoring levels based on BSI
    behavioral_signals.py    # 19 entropy/burst/counterparty/amount signals
    risk_scorer.py           # XGBoost over 70 features + SHAP

  graph_engine/
    graph_core.py            # NetworkX: circular flows, funnel hubs, layering
    graph_visualization.py   # NetworkX → Plotly interactive network diagrams
    
  sar/
    counterfactual.py        # "What-if" sensitivity analysis
    rag_service.py           # ChromaDB + Ollama embeddings for SAR retrieval
    sar_generator.py         # Mistral 7B narrative generation via Ollama
    sar_fallback.py          # Template-based SAR when Ollama unavailable
    compliance.py            # Automated compliance checklist
    audit.py                 # SQLite audit trail + data lineage
```

## Key Features

- **Behavioral Signals**: Entropy drift, temporal bursts, counterparty expansion, Benford's law
- **Graph Intelligence**: Circular flow detection, funnel/mule hubs, layering depth
- **BSI Score**: 0-100 composite stability index across 5 dimensions
- **Gray-Area Customers**: 50 false-positive-prone profiles (realistic for judges)
- **SHAP Explainability**: Per-alert feature attribution for regulatory transparency
- **Counterfactual Analysis**: "If this dimension were normal, risk drops by X%"
- **Fallback SAR Mode**: Works without Ollama using structured templates

## LLM Setup (Optional)

For AI-generated SAR narratives, install Ollama:
```bash
# Install Ollama from https://ollama.ai
ollama pull mistral:7b
ollama pull nomic-embed-text
```
Without Ollama, the system uses template-based fallback narratives.

## Cross-Validation with IBM AMLSim

We validated ASTRAS against IBM's open-source AMLSim dataset (20,000 accounts, 120K transactions, 1,804 fraud accounts) to prove our behavioral engine generalizes to external data.

### Run Validation
```bash
python validation/run_amlsim_validation.py
```

### Results (BSI-only, no retraining)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.89 |
| Mean BSI (fraud) | 75.9 |
| Mean BSI (legit) | 91.9 |
| BSI separation | +16 points |
| TPR at optimal threshold | 76% |
| FPR at optimal threshold | 7.4% |

| Dataset | AUC-ROC |
|---------|---------|
| ASTRAS synthetic data | 0.9986 |
| IBM AMLSim (external, zero retraining) | 0.89 |

The behavioral signals (entropy drift, burstiness, counterparty expansion, Benford's deviation) all show higher values for fraud accounts, confirming that ASTRAS detects behavioral anomalies independent of the data source.

## Tech Stack

Python, Streamlit, FastAPI, XGBoost, SHAP, NetworkX, Plotly, LangChain, ChromaDB, Ollama/Mistral 7B, SQLite
