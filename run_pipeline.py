"""
ASTRAS Full Pipeline Runner
Executes all phases: data generation → behavioral signals → graph analysis →
BSI → adaptive monitoring → risk scoring → RAG setup
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_full_pipeline():
    start = time.time()
    print("=" * 70)
    print("  ASTRAS - SAR Narrative Generator with Audit Trail")
    print("  Behavioral Intelligence-Driven AML Detection Pipeline")
    print("=" * 70)

    # # Phase 1: Data Generation
    # print("\n" + "=" * 70)
    # print("PHASE 1: Synthetic Data Generation")
    # print("=" * 70)
    # from services.data_generation.data_generator import run as generate_data
    # customers_df, transactions_df, alerts_df = generate_data()

    # If data is already generated 

    customers_df = pd.read_csv("data/generated/customers.csv")
    transactions_df = pd.read_csv("data/generated/transactions.csv")
    alerts_df = pd.read_csv("data/generated/alerts.csv")

    # Phase 2: Behavioral Signal Computation
    print("\n" + "=" * 70)
    print("PHASE 2: Behavioral Signal Computation")
    print("=" * 70)
    from services.behavioral_engine.behavioral_signals import compute_signals_for_all_customers
    behavioral_df = compute_signals_for_all_customers(customers_df, transactions_df)
    behavioral_df.to_csv('behavioral_signals.csv', index=False)
    print(f"  Behavioral signals: {len(behavioral_df.columns) - 1} features for {len(behavioral_df)} customers")

    # Phase 3: Graph Analysis
    print("\n" + "=" * 70)
    print("PHASE 3: NetworkX Graph Analysis")
    print("=" * 70)
    from services.graph_engine.graph_core import compute_graph_signals_for_all
    graph_df = compute_graph_signals_for_all(customers_df, transactions_df)
    graph_df.to_csv('graph_signals.csv', index=False)
    print(f"  Graph signals: {len(graph_df.columns) - 1} features for {len(graph_df)} customers")

    # Phase 4: BSI Computation
    print("\n" + "=" * 70)
    print("PHASE 4: Behavioral Stability Index (BSI)")
    print("=" * 70)
    from services.behavioral_engine.bsi import compute_bsi_for_all
    bsi_df = compute_bsi_for_all(behavioral_df, graph_df)
    bsi_df.to_csv('bsi_scores.csv', index=False)

    # Phase 5: Adaptive Monitoring
    print("\n" + "=" * 70)
    print("PHASE 5: Adaptive Monitoring System")
    print("=" * 70)
    from services.behavioral_engine.adaptive_monitor import compute_monitoring_states
    monitor_df = compute_monitoring_states(bsi_df)
    monitor_df.to_csv('monitoring_states.csv', index=False)

    # Phase 6: Risk Scoring (XGBoost Meta-Classifier)
    print("\n" + "=" * 70)
    print("PHASE 6: XGBoost Meta-Risk Classifier")
    print("=" * 70)
    from services.behavioral_engine.risk_scorer import run as run_risk_scorer
    features_df, model, explainer, feature_cols, enriched_alerts = run_risk_scorer(
        customers_df, transactions_df, alerts_df, behavioral_df, graph_df, bsi_df
    )

    # Phase 7: RAG Setup
    print("\n" + "=" * 70)
    print("PHASE 7: RAG Pipeline Setup")
    print("=" * 70)
    from services.sar.rag_service import save_templates
    save_templates()
    print("  (Vector store will be created on first SAR generation or can be run separately)")

    # Phase 8: Audit Tables
    print("\n" + "=" * 70)
    print("PHASE 8: Audit Trail Setup")
    print("=" * 70)
    from services.sar.audit import create_audit_tables
    create_audit_tables()
    print("  [OK] Audit tables created")

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print(f"  Total time: {elapsed:.1f} seconds")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Set up RAG vector store: python -c \"from services.sar.rag_service import setup; setup()\"")
    print("  2. Launch dashboard:        streamlit run app.py")
    print("  3. Launch API:              uvicorn api:app --reload")

    return {
        'customers': len(customers_df),
        'transactions': len(transactions_df),
        'alerts': len(alerts_df),
        'features': len(feature_cols),
        'elapsed_seconds': elapsed,
    }


if __name__ == "__main__":
    run_full_pipeline()

