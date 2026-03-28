import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load data and models
features_df = pd.read_csv('/Users/rishi/Documents/Barclays/Astras-AML-Engine/features.csv')
model = joblib.load("/Users/rishi/Documents/Barclays/Astras-AML-Engine/models/risk_classifier.pkl")
explainer = joblib.load("/Users/rishi/Documents/Barclays/Astras-AML-Engine/models/shap_explainer.pkl")

# Prepare features
exclude_cols = ['customer_id', 'is_suspicious', 'typology', 'risk_score', 'drift_level', 'num_unique_counterparties']
bool_cols = features_df.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    features_df[col] = features_df[col].astype(int)



feature_cols = [c for c in features_df.columns if c not in exclude_cols]
for col in feature_cols:
    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
features_df[feature_cols] = features_df[feature_cols].fillna(0)

X = features_df[feature_cols].values
y = features_df['is_suspicious'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


print("X_test shape:", X_test.shape)
print("Model expects:", model.n_features_in_)
# Compute SHAP values
shap_values = explainer.shap_values(X_test)
shap_importance = np.abs(shap_values).mean(axis=0)

# Normalize so all feature weights sum to 1
weight_lookup = dict(zip(feature_cols, shap_importance / shap_importance.sum()))

# BSI category mapping
bsi_mapping = {
    'entropy':      ['amount_entropy_drift', 'timing_entropy_drift', 'counterparty_entropy_drift',
                     'mean_amount_entropy', 'mean_timing_entropy', 'mean_counterparty_entropy'],
    'temporal':     ['burstiness_index', 'burst_events', 'inter_arrival_variance',
                     'max_daily_count', 'mean_daily_count'],
    'counterparty': ['counterparty_expansion_rate','counterparty_novelty_slope','total_unique_counterparties'],
    'amount':       ['benford_deviation', 'structuring_score', 'round_amount_ratio',
                     'amount_skewness', 'amount_kurtosis'],
    'network':      ['funnel_ratio', 'layer_depth', 'flow_velocity', 'has_circular_flow',
                     'rapid_passthrough_count', 'is_funnel_hub', 'pagerank_score']
}

# Sum weights per category
category_scores = {}
for cat, feats in bsi_mapping.items():
    matched_weights = [weight_lookup[f] for f in feats if f in weight_lookup]
    category_scores[cat] = np.mean(matched_weights) if matched_weights else 0.0

# Redistribute to sum to 1
mapped_total = sum(category_scores.values())
category_scores = {cat: score / mapped_total for cat, score in category_scores.items()}

# Print results
print("=== BSI Category SHAP Weights (sum to 1) ===")
for cat, score in sorted(category_scores.items(), key=lambda x: -x[1]):
    print(f"{cat:>15}: {score:.6f}")

print(f"\n{'TOTAL':>15}: {sum(category_scores.values()):.6f}")