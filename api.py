"""
ASTRAS FastAPI Backend
Central orchestration layer managing all services.
"""
import os
import sys
import json
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, MODELS_DIR, OUTPUTS_DIR, CHROMA_DIR, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

app = FastAPI(
    title="ASTRAS - SAR Narrative Generator",
    description="Behavioral Intelligence-Driven AML Detection & SAR Generation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy-loaded globals ──────────────────────────────────────────────
_state = {}


def _get_state():
    if not _state:
        features_df = pd.read_csv(os.path.join(os.path.dirname(MODELS_DIR), 'alert_features_scored.csv'))
        explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
        model = joblib.load(os.path.join(MODELS_DIR, 'risk_classifier.pkl'))

        # Load BSI if available
        bsi_path = os.path.join(os.path.dirname(MODELS_DIR), 'bsi_scores.csv')
        bsi_df = pd.read_csv(bsi_path) if os.path.exists(bsi_path) else None

        # Load monitoring states if available
        monitor_path = os.path.join(os.path.dirname(MODELS_DIR), 'monitoring_states.csv')
        monitor_df = pd.read_csv(monitor_path) if os.path.exists(monitor_path) else None

        _state.update({
            'features_df': features_df,
            'explainer': explainer,
            'feature_cols': feature_cols,
            'model': model,
            'bsi_df': bsi_df,
            'monitor_df': monitor_df,
        })
    return _state


def _get_llm_components():
    if 'vectorstore' not in _state:
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaEmbeddings, OllamaLLM

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_predict=LLM_MAX_TOKENS)

        _state['vectorstore'] = vectorstore
        _state['llm'] = llm
    return _state['vectorstore'], _state['llm']


# ── Models ────────────────────────────────────────────────────────────
class SARRequest(BaseModel):
    alert_id: int


class SARResponse(BaseModel):
    narrative: str
    compliance: dict
    audit_trail: dict


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "ASTRAS", "version": "2.0.0", "status": "running"}


@app.get("/api/alerts")
def get_alerts():
    conn = sqlite3.connect(DB_PATH)
    alerts = pd.read_sql("SELECT * FROM alerts", conn)
    customers = pd.read_sql("SELECT customer_id, name FROM customers", conn)
    conn.close()

    merged = alerts.merge(customers, on='customer_id', how='left')
    state = _get_state()
    if 'risk_score' not in merged.columns:
        scores = state['features_df'][['customer_id', 'risk_score']].drop_duplicates()
        merged = merged.merge(scores, on='customer_id', how='left')

    return merged.to_dict('records')


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: int):
    from services.sar_generator import load_alert_data
    try:
        alert, customer, txns = load_alert_data(alert_id)
        return {
            'alert': alert.to_dict(),
            'customer': customer.to_dict(),
            'num_transactions': len(txns),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/alerts/{alert_id}/behavioral")
def get_behavioral_signals(alert_id: int):
    from services.sar_generator import load_alert_data
    state = _get_state()

    alert, _, _ = load_alert_data(alert_id)
    cid = alert['customer_id']

    row = state['features_df'][state['features_df']['customer_id'] == cid]
    if len(row) == 0:
        raise HTTPException(404, "No behavioral data found")

    result = row.iloc[0].to_dict()

    if state['bsi_df'] is not None:
        bsi = state['bsi_df'][state['bsi_df']['customer_id'] == cid]
        if len(bsi) > 0:
            result['bsi'] = bsi.iloc[0].to_dict()

    if state['monitor_df'] is not None:
        mon = state['monitor_df'][state['monitor_df']['customer_id'] == cid]
        if len(mon) > 0:
            result['monitoring'] = mon.iloc[0].to_dict()

    return result


@app.get("/api/alerts/{alert_id}/counterfactual")
def get_counterfactual(alert_id: int):
    from services.sar_generator import load_alert_data
    from services.counterfactual import generate_counterfactual

    state = _get_state()
    alert, _, _ = load_alert_data(alert_id)

    result = generate_counterfactual(
        alert['customer_id'],
        state['features_df'],
        state['model'],
        state['feature_cols']
    )
    if result is None:
        raise HTTPException(404, "No data for counterfactual")
    return result


@app.post("/api/sar/generate")
def generate_sar(request: SARRequest):
    from services.sar_generator import generate_narrative, save_sar
    from services.compliance import validate_sar
    from services.audit import save_sar_record, create_audit_tables, load_alert_data

    state = _get_state()
    vectorstore, llm = _get_llm_components()

    try:
        create_audit_tables()

        narrative, audit_trail = generate_narrative(
            request.alert_id,
            vectorstore, llm,
            state['features_df'],
            state['explainer'],
            state['feature_cols'],
            state['bsi_df'],
        )

        compliance = validate_sar(narrative)
        sar_doc = save_sar(request.alert_id, narrative, audit_trail, compliance)

        alert_data = load_alert_data(request.alert_id)
        save_sar_record(
            request.alert_id,
            int(alert_data[1]['customer_id']),
            narrative,
            audit_trail,
            compliance
        )

        return {
            'narrative': narrative,
            'compliance': compliance,
            'audit_trail': {k: v for k, v in audit_trail.items() if k != 'narrative'},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/sars")
def list_sars():
    from services.audit import list_generated_sars
    return list_generated_sars()


@app.get("/api/sars/{alert_id}")
def get_sar(alert_id: int):
    from services.audit import load_sar_from_file
    doc = load_sar_from_file(alert_id)
    if doc is None:
        raise HTTPException(404, "SAR not found")
    return doc


@app.get("/api/system/info")
def system_info():
    state = _get_state()
    conn = sqlite3.connect(DB_PATH)
    counts = {
        'customers': pd.read_sql("SELECT COUNT(*) as c FROM customers", conn).iloc[0]['c'],
        'transactions': pd.read_sql("SELECT COUNT(*) as c FROM transactions", conn).iloc[0]['c'],
        'alerts': pd.read_sql("SELECT COUNT(*) as c FROM alerts", conn).iloc[0]['c'],
    }
    conn.close()

    return {
        'model': LLM_MODEL,
        'features': len(state['feature_cols']),
        'bsi_available': state['bsi_df'] is not None,
        **counts,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
