import pandas as pd
import numpy as np
import shap
import joblib
import os
import sys

# Paths
ROOT = '/Users/rishi/Documents/Barclays/Astras-AML-Engine'
os.chdir(ROOT)

try:
    # Load artifacts
    explainer = joblib.load('models/shap_explainer.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')
    features_df = pd.read_csv('alert_features_scored.csv')

    # Prep data
    bool_cols = features_df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)

    # Some features may have missing values
    features_df = features_df.fillna(0)
    X = features_df[feature_cols].values

    # Calculate SHAP values
    print("Calculating SHAP values... This may take a moment.")
    shap_values_obj = explainer.shap_values(X)

    if isinstance(shap_values_obj, list):
        shap_values = shap_values_obj[1]
    else:
        shap_values = shap_values_obj

    # Calculate mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # BSI Dimensions mapping
    bsi_mapping = {
        'entropy': [
            'amount_entropy_drift', 'timing_entropy_drift', 'counterparty_entropy_drift',
            'mean_amount_entropy', 'mean_timing_entropy', 'mean_counterparty_entropy'
        ],
        'temporal': [
            'burstiness_index', 'burst_events', 'inter_arrival_variance',
            'max_daily_count', 'mean_daily_count'
        ],
        'counterparty': [
            'counterparty_expansion_rate', 'counterparty_novelty_slope', 'total_unique_counterparties'
        ],
        'amount': [
            'benford_deviation', 'structuring_score', 'round_amount_ratio',
            'amount_skewness', 'amount_kurtosis'
        ],
        'network': [
            'funnel_ratio', 'layer_depth', 'flow_velocity', 'has_circular_flow',
            'rapid_passthrough_count', 'is_funnel_hub', 'pagerank_score'
        ]
    }

    dim_importance = {'entropy': 0.0, 'temporal': 0.0, 'counterparty': 0.0, 'amount': 0.0, 'network': 0.0}

    for dim, feats in bsi_mapping.items():
        for f in feats:
            if f in feature_cols:
                idx = feature_cols.index(f)
                dim_importance[dim] += mean_abs_shap[idx]

    total_importance = sum(dim_importance.values())

    print("\n--- SHAP-BASED BSI WEIGHTS ---")
    if total_importance > 0:
        for dim, imp in dim_importance.items():
            weight = imp / total_importance
            print(f"{dim}={weight:.4f}")
    else:
        print("Total importance is 0, cannot calculate weights.")

except Exception as e:
    print(f"Error: {e}")
