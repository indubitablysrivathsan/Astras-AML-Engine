import joblib
import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
MODELS_DIR = "models"

def get_feature_contributions(customer_id, features_df):
    # 1. Load the saved artifacts
    model = joblib.load(os.path.join(MODELS_DIR, 'risk_classifier.pkl'))
    explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

    # 2. Get the specific customer's data
    customer_data = features_df[features_df['customer_id'] == customer_id][feature_cols]
    
    if customer_data.empty:
        return "Customer ID not found."

    # 3. Calculate SHAP values for this specific instance
    # Note: If using TreeExplainer, it returns a list for multiclass or a single array for binary
    shap_values = explainer.shap_values(customer_data)
    
    # 4. Create a DataFrame of contributions
    contributions = pd.DataFrame({
        'feature': feature_cols,
        'contribution': shap_values[0] if isinstance(shap_values, list) else shap_values.flatten(),
        'actual_value': customer_data.values.flatten()
    })
    
    return contributions.sort_values(by='contribution', ascending=False)

def validate_weights_with_shap(features_df):
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
    
    # Calculate global importance (mean absolute SHAP value)
    shap_values = explainer.shap_values(features_df[feature_cols])
    global_importance = np.abs(shap_values).mean(0)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'learned_importance': global_importance
    })

    # Define your clusters (adjust based on your actual feature names)
    clusters = {
        # Dimension 1: Entropy stability
        'Entropy_Stability': [
            'amount_entropy_drift', 'timing_entropy_drift', 'counterparty_entropy_drift',
            'mean_amount_entropy', 'mean_timing_entropy', 'mean_counterparty_entropy',
            'entropy_stability', 'bsi_score'
        ],
        
        # Dimension 2: Temporal regularity
        'Temporal_Regularity': [
            'burstiness_index', 'burst_events', 'max_daily_count', 
            'mean_daily_count', 'inter_arrival_variance', 'temporal_stability'
        ],
        
        # Dimension 3: Counterparty stability
        'Counterparty_Stability': [
            'counterparty_expansion_rate', 'counterparty_novelty_slope', 
            'total_unique_counterparties', 'num_unique_counterparties', 
            'counterparty_stability'
        ],
        
        # Dimension 4: Amount normality
        'Amount_Normality': [
            'benford_deviation', 'structuring_score', 'amount_skewness', 
            'amount_kurtosis', 'num_just_under_10k', 'pct_just_under_10k',
            'round_amount_ratio', 'amount_stability'
        ],
        
        # Dimension 5: Network topology stability
        'Network_Topology': [
            'funnel_ratio', 'layer_depth', 'flow_velocity', 'has_circular_flow',
            'num_cycles', 'max_cycle_length', 'cycle_total_amount', 'is_funnel_hub',
            'in_degree', 'out_degree', 'total_inflow', 'total_outflow',
            'reachable_nodes', 'rapid_passthrough_count', 'network_stability'
        ]
    }

    # Aggregate Learned Weights
    cluster_comparison = []
    for name, cols in clusters.items():
        # Sum of importance for features in this cluster
        total_imp = importance_df[importance_df['feature'].isin(cols)]['learned_importance'].sum()
        cluster_comparison.append({'dimension': name, 'learned_weight': total_imp})

    comparison_df = pd.DataFrame(cluster_comparison)
    
    # Normalize to 1.0 (to compare with hand-tuned percentages)
    comparison_df['normalized_learned_weight'] = comparison_df['learned_weight'] / comparison_df['learned_weight'].sum()

    return comparison_df

def main():
    # 1. Load the features data used for the model
    # (Assuming you saved your engineered features to a CSV)
    features_path = os.path.join(DATA_DIR, 'generated', 'features.csv')
    if not os.path.exists(features_path):
        print("Features file not found. Make sure you've run feature engineering first.")
        return

    features_df = pd.read_csv(features_path)

    # 2. Run Global Validation (The "Challenge" Part)
    print("Calculating Empirical Weights from SHAP...")
    comparison_df = validate_weights_with_shap(features_df)

    comparison_df.to_csv("weights.csv")
    
    print("\n--- DIMENSION WEIGHT COMPARISON ---")
    print(comparison_df[['dimension', 'normalized_learned_weight']])
    print("\n[SUCCESS] weights.csv has been generated.")

    # 3. Check specific High-Risk Customer contributions
    # Let's look at a customer with a high risk score
    # (Assuming 'risk_score' exists in your features_df)
    high_risk_sample = features_df.nlargest(1, 'risk_score')['customer_id'].values[0]
    
    print(f"\nAnalyzing Customer ID: {high_risk_sample}")
    contributions = get_feature_contributions(high_risk_sample, features_df)
    print(contributions.head(10)) # Top 10 factors pushing score UP

if __name__ == "__main__":
    main()