"""
Counterfactual SAR Simulation
Structured sensitivity analysis of behavioral risk components.
Identifies which dimensions drove the risk increase and whether
the account barely crossed the threshold.
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, HIGH_RISK_THRESHOLD


def load_model_artifacts():
    """Load trained model and feature columns."""
    model = joblib.load(os.path.join(MODELS_DIR, 'risk_classifier.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    return model, feature_cols


def generate_counterfactual(customer_id, features_df, model=None, feature_cols=None):
    """
    Generate counterfactual analysis: what would need to change to flip the SAR decision.
    Returns the current state and the counterfactual adjustments needed.
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model_artifacts()

    row = features_df[features_df['customer_id'] == customer_id]
    if len(row) == 0:
        return None

    row = row.iloc[0]
    X = row[feature_cols].values.reshape(1, -1)
    current_risk = model.predict_proba(X)[0][1]

    # Key behavioral dimensions to test
    dimensions = {
        'Entropy Drift': ['amount_entropy_drift', 'timing_entropy_drift', 'counterparty_entropy_drift'],
        'Temporal Burst': ['burstiness_index', 'burst_events', 'max_daily_count'],
        'Counterparty Expansion': ['counterparty_expansion_rate', 'counterparty_novelty_slope',
                                   'total_unique_counterparties'],
        'Amount Anomaly': ['benford_deviation', 'structuring_score', 'round_amount_ratio'],
        'Network Topology': ['funnel_ratio', 'layer_depth', 'flow_velocity'],
        'Transaction Volume': ['total_volume', 'total_transactions', 'volume_to_income_ratio'],
        'International Activity': ['num_intl_txns', 'num_high_risk_country_txns', 'pct_intl'],
        'Cash Activity': ['num_cash_txns', 'pct_cash', 'total_cash_amount'],
    }

    results = {
        'customer_id': customer_id,
        'current_risk_score': float(current_risk),
        'sar_triggered': current_risk >= HIGH_RISK_THRESHOLD,
        'threshold': HIGH_RISK_THRESHOLD,
        'dimension_impacts': [],
    }

    for dim_name, dim_features in dimensions.items():
        # Available features for this dimension
        available = [f for f in dim_features if f in feature_cols]
        if not available:
            continue

        # Create counterfactual: normalize this dimension's features
        X_cf = X.copy()
        for feat in available:
            idx = feature_cols.index(feat)
            # Set to population median (approximate "normal" behavior)
            col_values = features_df[feat].values
            median_val = np.median(col_values)
            X_cf[0][idx] = median_val

        cf_risk = model.predict_proba(X_cf)[0][1]
        risk_reduction = current_risk - cf_risk
        pct_contribution = (risk_reduction / current_risk * 100) if current_risk > 0 else 0

        # Current values for this dimension
        current_vals = {f: float(row[f]) for f in available}
        median_vals = {f: float(np.median(features_df[f].values)) for f in available}

        results['dimension_impacts'].append({
            'dimension': dim_name,
            'features': available,
            'current_values': current_vals,
            'counterfactual_values': median_vals,
            'risk_with_normalization': float(cf_risk),
            'risk_reduction': float(risk_reduction),
            'contribution_pct': float(pct_contribution),
            'would_flip_decision': cf_risk < HIGH_RISK_THRESHOLD and current_risk >= HIGH_RISK_THRESHOLD,
        })

    # Sort by contribution
    results['dimension_impacts'].sort(key=lambda x: x['contribution_pct'], reverse=True)

    # Top contributors
    total_contribution = sum(d['contribution_pct'] for d in results['dimension_impacts'])
    top_2 = results['dimension_impacts'][:2]
    top_2_contribution = sum(d['contribution_pct'] for d in top_2)

    results['summary'] = {
        'top_risk_dimensions': [d['dimension'] for d in top_2],
        'top_2_contribution_pct': round(top_2_contribution, 1),
        'barely_crossed_threshold': abs(current_risk - HIGH_RISK_THRESHOLD) < 0.1,
        'total_explained_pct': round(total_contribution, 1),
    }

    return results


def format_counterfactual_report(cf_result):
    """Format counterfactual analysis as human-readable report."""
    if cf_result is None:
        return "No data available for counterfactual analysis."

    lines = [
        "COUNTERFACTUAL SAR SIMULATION",
        "=" * 50,
        f"Customer ID: {cf_result['customer_id']}",
        f"Current Risk Score: {cf_result['current_risk_score']:.2%}",
        f"SAR Triggered: {'YES' if cf_result['sar_triggered'] else 'NO'}",
        f"Threshold: {cf_result['threshold']:.0%}",
        "",
        "DIMENSION ANALYSIS:",
        "-" * 40,
    ]

    for dim in cf_result['dimension_impacts']:
        status = "WOULD FLIP" if dim['would_flip_decision'] else ""
        lines.append(f"\n{dim['dimension']} [{dim['contribution_pct']:.1f}% of risk] {status}")
        lines.append(f"  Risk if normalized: {dim['risk_with_normalization']:.2%} "
                      f"(reduction: {dim['risk_reduction']:.2%})")
        for feat in dim['features']:
            curr = dim['current_values'].get(feat, 0)
            cf = dim['counterfactual_values'].get(feat, 0)
            lines.append(f"  - {feat}: {curr:.3f} → {cf:.3f}")

    summary = cf_result['summary']
    lines.extend([
        "",
        "SUMMARY:",
        f"Top drivers: {', '.join(summary['top_risk_dimensions'])}",
        f"These explain {summary['top_2_contribution_pct']:.0f}% of the risk",
        f"Barely crossed threshold: {'YES' if summary['barely_crossed_threshold'] else 'NO'}",
    ])

    return "\n".join(lines)
