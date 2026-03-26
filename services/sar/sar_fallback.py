"""
Fallback SAR Generation
Generates template-based SAR narratives when Ollama/LLM is unavailable.
Uses structured templates filled with actual data - no AI required.
"""
import pandas as pd
import numpy as np
from datetime import datetime


def generate_fallback_narrative(alert, customer, transactions, risk_score=0.0,
                                 bsi_data=None, shap_drivers=None):
    """Generate a structured SAR narrative without LLM."""
    txns = transactions.copy()
    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'])
    txns = txns.sort_values('transaction_date')

    deposits = txns[txns['transaction_type'] == 'deposit']
    withdrawals = txns[txns['transaction_type'] == 'withdrawal']
    cash = txns[txns['method'] == 'cash']
    wire = txns[txns['method'] == 'wire']
    intl = txns[txns['country'] != 'USA']

    total_deposits = deposits['amount'].sum()
    total_withdrawals = withdrawals['amount'].sum()
    date_start = txns['transaction_date'].min().strftime('%Y-%m-%d')
    date_end = txns['transaction_date'].max().strftime('%Y-%m-%d')
    typology = alert['alert_type'].replace('_', ' ').title()

    # BSI context
    bsi_text = ""
    if bsi_data:
        bsi_text = (f" The Behavioral Stability Index (BSI) for this account is "
                    f"{bsi_data.get('bsi_score', 'N/A')}/100, indicating "
                    f"{bsi_data.get('drift_level', 'unknown')} behavioral drift.")

    # SHAP context
    shap_text = ""
    if shap_drivers is not None and len(shap_drivers) > 0:
        top3 = shap_drivers.head(3)
        drivers = ", ".join([f"{r['feature'].replace('_',' ').title()} (impact: {r['shap_value']:+.3f})"
                             for _, r in top3.iterrows()])
        shap_text = f" Key risk drivers identified by SHAP analysis include: {drivers}."

    narrative = f"""INTRODUCTION
This Suspicious Activity Report (SAR) is filed regarding the account of {customer['name']}, who has been identified through automated behavioral analysis as engaging in activity consistent with {typology}. The account was flagged with a risk score of {risk_score:.1%} by the institution's AI-assisted monitoring system.{bsi_text}

WHO
{customer['name']} is a {customer['customer_type']} customer with {('SSN' if customer['customer_type'] == 'individual' else 'EIN')} {customer['ssn_ein']}. The subject's stated occupation is {customer['occupation']}, with a reported annual income of ${customer['annual_income']:,.2f}. The account was opened on {customer['account_open_date']} and is located at {customer['address']}, {customer['city']}, {customer['state']} {customer['zip_code']}. The account carries a risk rating of {customer['risk_rating'].title()}.

WHAT
Between {date_start} and {date_end}, the account processed {len(txns)} transactions totaling ${txns['amount'].sum():,.2f}. This included {len(deposits)} deposits totaling ${total_deposits:,.2f} and {len(withdrawals)} withdrawals totaling ${total_withdrawals:,.2f}, resulting in a net flow of ${total_deposits - total_withdrawals:,.2f}. Of these transactions, {len(cash)} were cash transactions totaling ${cash['amount'].sum():,.2f}, and {len(wire)} were wire transfers. {len(intl)} transactions involved international counterparties across {intl['country'].nunique()} countries{(', including ' + ', '.join(intl['country'].unique()[:5])) if len(intl) > 0 else ''}.

WHEN
The suspicious activity commenced on {date_start} and continued through {date_end}, spanning a period of {(txns['transaction_date'].max() - txns['transaction_date'].min()).days} days. The average transaction frequency was approximately {len(txns) / max((txns['transaction_date'].max() - txns['transaction_date'].min()).days, 1) * 7:.1f} transactions per week.

WHERE
Transactions were conducted across {txns['location'].nunique()} distinct locations. {'International jurisdictions involved include ' + ', '.join(intl['country'].unique()[:8]) + '.' if len(intl) > 0 else 'All transactions were domestic.'}

WHY SUSPICIOUS
The account activity is consistent with {typology} based on multiple behavioral indicators.{shap_text} The transaction patterns deviate significantly from expected behavior for a customer with the subject's profile and stated income. The volume-to-income ratio of {txns['amount'].sum() / max(customer['annual_income'], 1):.1f}x indicates transaction activity disproportionate to the subject's reported financial capacity.{bsi_text}

HOW
The subject appears to be utilizing the account to {_get_how_text(alert['alert_type'], len(cash), len(wire), len(intl))}. This pattern was detected through the institution's behavioral intelligence system, which monitors entropy drift, temporal burst patterns, counterparty network expansion, and amount distribution anomalies.

CONCLUSION
Based on the behavioral analysis, risk scoring ({risk_score:.1%}), and pattern recognition described above, this activity is being reported as suspected {typology}. The institution's AI-assisted system generated this narrative for analyst review. No autonomous filing has occurred — this SAR is submitted under human oversight and approval. The account will continue to be monitored under enhanced surveillance protocols."""

    return narrative.strip()


def _get_how_text(typology, num_cash, num_wire, num_intl):
    """Get typology-specific HOW section text."""
    texts = {
        'structuring': f"structure cash deposits below Currency Transaction Report thresholds, with {num_cash} cash transactions designed to avoid detection",
        'rapid_movement': f"rapidly move funds through the account as a conduit, with {num_wire} wire transfers showing minimal retention time between deposits and withdrawals",
        'layering': f"layer funds through multiple transfer chains across jurisdictions, with {num_wire} wire transfers creating distance from the original source of funds",
        'trade_based': f"disguise value transfers as trade payments, with invoiced amounts significantly exceeding fair market values for the described goods",
        'cash_intensive': f"commingle potentially illicit cash with business deposits, with {num_cash} cash deposits significantly exceeding industry benchmarks",
        'shell_company': f"facilitate financial transactions through an entity with no verifiable business operations, using {num_wire} wire transfers to offshore jurisdictions",
        'funnel_account': f"aggregate funds from multiple sources before redistributing to a limited number of beneficiaries, exhibiting classic funnel account behavior",
        'third_party_payments': f"receive and redistribute funds through third-party relationships with no apparent legitimate purpose",
        'round_tripping': f"cycle funds through foreign entities to disguise their origin, with {num_intl} international transactions creating an appearance of legitimate investment returns",
        'smurfing': f"coordinate deposits by multiple individuals to avoid reporting thresholds, with {num_cash} cash deposits distributed across multiple branch locations",
    }
    return texts.get(typology, f"conduct transactions inconsistent with the stated account purpose, involving {num_wire} wire transfers and {num_cash} cash transactions")


def is_ollama_available():
    """Check if Ollama is running and the model is available."""
    try:
        import urllib.request
        req = urllib.request.Request('http://localhost:11434/api/tags', method='GET')
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False
