import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

import os
ROOT = os.path.dirname(os.path.abspath(__file__))

features_df = pd.read_csv(os.path.join(ROOT, 'features.csv'))
model = joblib.load(os.path.join(ROOT, 'models', 'risk_classifier.pkl'))
explainer = joblib.load(os.path.join(ROOT, 'models', 'shap_explainer.pkl'))
feature_cols = joblib.load(os.path.join(ROOT, 'models', 'feature_columns.pkl'))

exclude_cols = ['customer_id', 'is_suspicious', 'typology', 'risk_score', 'drift_level']
bool_cols = features_df.select_dtypes(include=['bool']).columns.tolist()

for col in bool_cols:
    features_df[col] = features_df[col].astype(int)

feature_cols = [c for c in features_df.columns if c not in exclude_cols]
X = features_df[feature_cols].values
y = features_df['is_suspicious'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

shap_values_obj = explainer.shap_values(X_test)
if isinstance(shap_values_obj, list):
    shap_values = shap_values_obj[1]
else:
    shap_values = shap_values_obj

shap_importance = np.abs(shap_values).mean(axis=0)

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

dim_importance_sum = {'entropy': 0.0, 'temporal': 0.0, 'counterparty': 0.0, 'amount': 0.0, 'network': 0.0}
dim_count = {'entropy': 0, 'temporal': 0, 'counterparty': 0, 'amount': 0, 'network': 0}

for dim, feats in bsi_mapping.items():
    for f in feats:
        if f in feature_cols:
            idx = feature_cols.index(f)
            dim_importance_sum[dim] += shap_importance[idx]
            dim_count[dim] += 1

print("\n--- DIMENSION AVERAGE SHAP WEIGHT ---")
for dim in dim_importance_sum.keys():
    if dim_count[dim] > 0:
        avg_weight = dim_importance_sum[dim] / dim_count[dim]
        print(f"{dim.capitalize()}: {avg_weight:.6f}")
    else:
        print(f"{dim.capitalize()}: 0.000000")

print("\n--- DIMENSION TOTAL SHAP WEIGHT ---")
total_sum = sum(dim_importance_sum.values())
if total_sum > 0:
    for dim in dim_importance_sum.keys():
        pct = (dim_importance_sum[dim] / total_sum) * 100
        print(f"{dim.capitalize()}: {pct:.2f}%")

