"""
SAR Narrative Generation Service
Generates FinCEN-compliant SAR narratives using Mistral 7B with RAG context,
behavioral signals, and full audit trail.
"""
import pandas as pd
import numpy as np
import sqlite3
import json
import shap
import joblib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, MODELS_DIR, OUTPUTS_DIR


def load_alert_data(alert_id, db_path=DB_PATH):
    """Load all data for a specific alert."""
    conn = sqlite3.connect(db_path)
    alert = pd.read_sql(f"SELECT * FROM alerts WHERE alert_id = {int(alert_id)}", conn).iloc[0]
    customer = pd.read_sql(f"SELECT * FROM customers WHERE customer_id = {int(alert['customer_id'])}", conn).iloc[0]
    transactions = pd.read_sql(f"SELECT * FROM transactions WHERE customer_id = {int(alert['customer_id'])}", conn)
    conn.close()
    return alert, customer, transactions


def format_customer_info(customer):
    """Format customer information for narrative."""
    ctype = 'SSN' if customer['customer_type'] == 'individual' else 'EIN'
    return f"""Customer Name: {customer['name']}
Customer Type: {customer['customer_type'].title()}
{ctype}: {customer['ssn_ein']}
Occupation/Business: {customer['occupation']}
Address: {customer['address']}, {customer['city']}, {customer['state']} {customer['zip_code']}
Annual Income: ${customer['annual_income']:,.2f}
Account Opening Date: {customer['account_open_date']}
Risk Rating: {customer['risk_rating'].title()}"""


def format_transaction_summary(transactions, top_n=20):
    """Format transaction summary."""
    txns = transactions.copy()
    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'])
    txns = txns.sort_values('transaction_date', ascending=False)

    deposits = txns[txns['transaction_type'] == 'deposit']
    withdrawals = txns[txns['transaction_type'] == 'withdrawal']
    cash = txns[txns['method'] == 'cash']
    intl = txns[txns['country'] != 'USA']

    crypto = txns[txns['method'] == 'crypto_exchange'] if 'method' in txns.columns else pd.DataFrame()
    currencies_used = sorted(txns['currency'].unique().tolist()) if 'currency' in txns.columns else ['USD']
    crypto_assets = sorted(txns['crypto_asset'].dropna().unique().tolist()) if 'crypto_asset' in txns.columns else []

    summary = f"""Total Transactions: {len(txns)}
Date Range: {txns['transaction_date'].min().strftime('%Y-%m-%d')} to {txns['transaction_date'].max().strftime('%Y-%m-%d')}
Total Deposits: ${deposits['amount'].sum():,.2f} ({len(deposits)} transactions)
Total Withdrawals: ${withdrawals['amount'].sum():,.2f} ({len(withdrawals)} transactions)
Net Flow: ${deposits['amount'].sum() - withdrawals['amount'].sum():,.2f}
Cash Transactions: {len(cash)} totaling ${cash['amount'].sum():,.2f}
International Transactions: {len(intl)} across {intl['country'].nunique()} countries
Currencies Used: {', '.join(currencies_used)}"""

    if len(crypto) > 0:
        summary += f"\nCryptocurrency Exchange Transactions: {len(crypto)} totaling ${crypto['amount'].sum():,.2f}"
        summary += f"\nCrypto Assets Involved: {', '.join(crypto_assets) if crypto_assets else 'N/A'}"

    if len(intl) > 0:
        summary += f"\nCountries: {', '.join(intl['country'].unique()[:10])}"

    summary += f"\n\nTop {min(top_n, len(txns))} Transactions:"
    for _, txn in txns.head(top_n).iterrows():
        line = f"\n- {txn['transaction_date'].strftime('%Y-%m-%d')}: {txn['transaction_type'].title()} ${txn['amount']:,.2f} via {txn['method']}"
        if pd.notna(txn.get('counterparty')) and txn['counterparty']:
            line += f" (Counterparty: {txn['counterparty']}, Country: {txn['country']})"
        summary += line

    return summary


def format_behavioral_context(customer_id, features_df, bsi_df=None):
    """Format behavioral intelligence context for the prompt."""
    row = features_df[features_df['customer_id'] == customer_id]
    if len(row) == 0:
        return "No behavioral signals available."

    row = row.iloc[0]
    context = "Behavioral Intelligence Summary:"

    # BSI
    if bsi_df is not None:
        bsi_row = bsi_df[bsi_df['customer_id'] == customer_id]
        if len(bsi_row) > 0:
            bsi = bsi_row.iloc[0]
            context += f"\n- Behavioral Stability Index: {bsi.get('bsi_score', 'N/A')}/100 ({bsi.get('drift_level', 'N/A')})"
            context += f"\n  - Entropy Stability: {bsi.get('entropy_stability', 'N/A')}%"
            context += f"\n  - Temporal Stability: {bsi.get('temporal_stability', 'N/A')}%"
            context += f"\n  - Counterparty Stability: {bsi.get('counterparty_stability', 'N/A')}%"
            context += f"\n  - Amount Stability: {bsi.get('amount_stability', 'N/A')}%"
            context += f"\n  - Network Stability: {bsi.get('network_stability', 'N/A')}%"

    # Key behavioral signals
    if 'burstiness_index' in row:
        context += f"\n- Burstiness Index: {row.get('burstiness_index', 0):.3f}"
    if 'amount_entropy_drift' in row:
        context += f"\n- Entropy Drift: {row.get('amount_entropy_drift', 0):.3f}"
    if 'structuring_score' in row:
        context += f"\n- Structuring Score: {row.get('structuring_score', 0):.3f}"
    if 'counterparty_expansion_rate' in row:
        context += f"\n- Counterparty Expansion Rate: {row.get('counterparty_expansion_rate', 0):.2f}"
    if 'flow_velocity' in row:
        context += f"\n- Flow Velocity: {row.get('flow_velocity', 0):.4f}"
    if 'funnel_ratio' in row:
        context += f"\n- Funnel Ratio: {row.get('funnel_ratio', 1):.2f}"
    if 'layer_depth' in row:
        context += f"\n- Layer Depth: {int(row.get('layer_depth', 0))}"

    return context


def get_shap_drivers(alert, features_df, explainer, feature_cols):
    """Get top SHAP risk drivers for an alert."""
    row = features_df[features_df['customer_id'] == alert['customer_id']]
    if len(row) == 0:
        return pd.DataFrame(), 0.0

    row = row.iloc[0]
    X = row[feature_cols].values.reshape(1, -1)
    shap_vals = explainer.shap_values(X)

    drivers = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_vals[0]
    }).sort_values('shap_value', ascending=False)

    risk_score = row.get('risk_score', 0.0)
    return drivers.head(10), risk_score


def generate_narrative(alert_id, vectorstore, llm, features_df, explainer,
                       feature_cols, bsi_df=None, db_path=DB_PATH):
    """Generate complete SAR narrative with audit trail."""
    print(f"\n{'=' * 60}")
    print(f"GENERATING SAR FOR ALERT {alert_id}")
    print(f"{'=' * 60}")

    alert, customer, transactions = load_alert_data(alert_id, db_path)
    customer_info = format_customer_info(customer)
    txn_summary = format_transaction_summary(transactions)
    behavioral_context = format_behavioral_context(alert['customer_id'], features_df, bsi_df)
    top_drivers, risk_score = get_shap_drivers(alert, features_df, explainer, feature_cols)

    # Format SHAP drivers
    shap_text = "Top Risk Drivers (SHAP Analysis):"
    for _, d in top_drivers.head(5).iterrows():
        feat_name = d['feature'].replace('_', ' ').title()
        shap_text += f"\n- {feat_name}: SHAP impact {d['shap_value']:+.3f}"

    # RAG retrieval
    query = f"{alert['alert_type'].replace('_', ' ')} suspicious activity"
    retrieved = vectorstore.similarity_search(query, k=2)
    context = "\n\n---\n\n".join([
        f"Template ({doc.metadata['typology']}):\n{doc.page_content[:800]}"
        for doc in retrieved
    ])

    is_crypto = 'crypto' in str(alert['alert_type']).lower()

    prompt = f"""You are a compliance officer writing a Suspicious Activity Report (SAR) narrative for FinCEN.

Use these SAR template examples as guidance for structure and tone:
{context}

Based on the customer information, transaction data, and behavioral intelligence below, write a complete SAR narrative.

Customer Information:
{customer_info}

Transaction Summary:
{txn_summary}

Alert Details:
Alert ID: {alert['alert_id']}
Alert Type: {alert['alert_type'].replace('_', ' ').title()}
Alert Date: {alert['alert_date']}
Severity: {alert['severity'].title()}
Risk Score: {risk_score:.2%}

{shap_text}

{behavioral_context}

CRITICAL REQUIREMENTS:
1. Use ONLY the information provided above — do not fabricate details.
2. Follow this EXACT section order (each as a heading on its own line):
   INTRODUCTION
   {"SOURCE OF FIAT FUNDS" if is_crypto else ""}
   WHO
   WHAT
   WHEN
   WHERE
   WHY SUSPICIOUS
   HOW
   CONCLUSION
   All sections are MANDATORY. Do NOT skip HOW or CONCLUSION.
3. HOW section: describe the specific mechanism used — how funds moved, methods employed (wire, crypto exchange, cash), how layering/structuring was executed, and how detection was evaded.
4. SOURCE OF FIAT FUNDS (crypto cases only): trace fiat deposits that preceded crypto purchases per FinCEN FIN-2019-G001.
5. Reference behavioral intelligence findings (BSI score, entropy drift, burstiness) in WHY SUSPICIOUS.
6. Use specific amounts, dates, and counterparty names from the data.
7. Length: 700-1000 words total across all sections.
8. Start directly with "INTRODUCTION" — no preamble or meta-commentary.

SAR NARRATIVE:
"""

    print(f"Generating narrative with {LLM_MODEL}...")
    narrative = llm.invoke(prompt).strip()
    print(f"[OK] {len(narrative.split())} words generated")

    # Build audit trail
    audit_trail = {
        'alert_id': int(alert_id),
        'generation_timestamp': datetime.now().isoformat(),
        'model': LLM_MODEL,
        'temperature': LLM_TEMPERATURE,
        'customer_id': int(customer['customer_id']),
        'customer_name': customer['name'],
        'num_transactions_analyzed': len(transactions),
        'transaction_date_range': f"{transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}",
        'risk_score': float(risk_score),
        'typology_detected': alert['alert_type'],
        'shap_top_features': top_drivers.head(5).to_dict('records') if len(top_drivers) > 0 else [],
        'templates_retrieved': [
            {'typology': doc.metadata['typology'], 'template_id': doc.metadata.get('template_id', 'unknown')}
            for doc in retrieved
        ],
        'generation_params': {'temperature': LLM_TEMPERATURE, 'max_tokens': LLM_MAX_TOKENS},
        'narrative': narrative,
        'narrative_word_count': len(narrative.split()),
        'status': 'draft',
        'reviewed_by': None,
        'approved_by': None,
        'filed_date': None,
    }

    # Add BSI to audit trail
    if bsi_df is not None:
        bsi_row = bsi_df[bsi_df['customer_id'] == alert['customer_id']]
        if len(bsi_row) > 0:
            bsi = bsi_row.iloc[0]
            audit_trail['bsi_score'] = float(bsi.get('bsi_score', 0))
            audit_trail['drift_level'] = bsi.get('drift_level', 'unknown')

    return narrative, audit_trail


def save_sar(alert_id, narrative, audit_trail, compliance_check):
    """Save SAR narrative and audit trail to disk."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    sar_doc = {
        'alert_id': alert_id,
        'narrative': narrative,
        'audit_trail': audit_trail,
        'compliance_check': compliance_check,
        'generated_at': datetime.now().isoformat()
    }

    json_path = os.path.join(OUTPUTS_DIR, f'sar_alert_{alert_id}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sar_doc, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(OUTPUTS_DIR, f'sar_narrative_{alert_id}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"SUSPICIOUS ACTIVITY REPORT - NARRATIVE\n")
        f.write(f"Alert ID: {alert_id}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(narrative)

    print(f"[OK] SAR saved: {json_path}")
    return sar_doc
