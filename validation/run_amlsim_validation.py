"""
Cross-Validation with IBM AMLSim
Loads IBM's AMLSim synthetic data, maps it to ASTRAS schema,
and runs our behavioral intelligence pipeline to validate detection.
"""
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from services.behavioral_signals import compute_all_signals
from services.graph_engine import build_transaction_graph, compute_graph_signals
from services.bsi import compute_bsi

AMLSIM_DIR = os.path.join(SCRIPT_DIR, 'amlsim', '20K_fanin200cycle200')

print("=" * 70)
print("  ASTRAS x IBM AMLSim Cross-Validation")
print("  Testing ASTRAS behavioral detection on external AML data")
print("=" * 70)

# ─── PHASE 1: Load & Map IBM AMLSim Data ─────────────────────────────
print("\n" + "=" * 70)
print("PHASE 1: Loading IBM AMLSim Data")
print("=" * 70)

nodes = pd.read_csv(os.path.join(AMLSIM_DIR, 'nodes.csv'))
txns_raw = pd.read_csv(os.path.join(AMLSIM_DIR, 'transactions.csv'))

print(f"  Accounts: {len(nodes):,} ({nodes['isFraud'].sum():,} fraudulent, "
      f"{(nodes['isFraud'].mean()*100):.1f}%)")
print(f"  Transactions: {len(txns_raw):,}")
print(f"  Typologies: fan-in (200 patterns) + cycle (200 patterns)")

# Map to ASTRAS schema
# AMLSim 'time' is integer step (1-149), map to actual dates
base_date = datetime(2025, 1, 1)
txns = pd.DataFrame({
    'transaction_id': range(len(txns_raw)),
    'customer_id': txns_raw['sourceNodeId'].astype(str),
    'counterparty': txns_raw['targetNodeId'].astype(str),
    'amount': txns_raw['value'],
    'transaction_date': [base_date + timedelta(days=int(t * 2.5))
                         for t in txns_raw['time']],
    # AMLSim doesn't have these — synthesize realistically
    'transaction_type': np.where(np.random.random(len(txns_raw)) > 0.5,
                                  'deposit', 'withdrawal'),
    'method': np.random.choice(['wire', 'ach', 'check', 'cash'],
                                size=len(txns_raw),
                                p=[0.35, 0.30, 0.20, 0.15]),
    'country': np.random.choice(['USA', 'USA', 'USA', 'USA', 'UK', 'China',
                                  'UAE', 'Panama', 'Cayman Islands'],
                                 size=len(txns_raw)),
    'location': 'branch_' + pd.Series(np.random.randint(1, 20, len(txns_raw))).astype(str),
})

# Build ground truth mapping
fraud_ids = set(nodes[nodes['isFraud'] == 1]['nodeid'].astype(str))
print(f"  Fraud account IDs: {len(fraud_ids)}")

# ─── PHASE 2: Sample for Tractable Analysis ──────────────────────────
print("\n" + "=" * 70)
print("PHASE 2: Sampling Accounts for Analysis")
print("=" * 70)

# Take all fraud accounts + random sample of legit accounts
# (running 20K accounts would take hours)
fraud_accounts = list(fraud_ids)
legit_accounts = list(set(nodes[nodes['isFraud'] == 0]['nodeid'].astype(str)))
np.random.seed(42)
sampled_legit = list(np.random.choice(legit_accounts,
                                       size=min(500, len(legit_accounts)),
                                       replace=False))

# Take a subset of fraud too if there are too many
sampled_fraud = list(np.random.choice(fraud_accounts,
                                       size=min(200, len(fraud_accounts)),
                                       replace=False))

sampled_ids = set(sampled_fraud + sampled_legit)
sampled_txns = txns[txns['customer_id'].isin(sampled_ids)].copy()

print(f"  Sampled accounts: {len(sampled_ids)} "
      f"({len(sampled_fraud)} fraud + {len(sampled_legit)} legit)")
print(f"  Sampled transactions: {len(sampled_txns):,}")

# ─── PHASE 3: Compute Behavioral Signals ─────────────────────────────
print("\n" + "=" * 70)
print("PHASE 3: Computing ASTRAS Behavioral Signals")
print("=" * 70)

start_time = time.time()
all_signals = []
customer_list = sorted(sampled_ids)

for i, cid in enumerate(customer_list):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1} / {len(customer_list)} customers")
    ctxns = sampled_txns[sampled_txns['customer_id'] == cid]
    if len(ctxns) < 3:
        continue
    signals = compute_all_signals(cid, ctxns)
    signals['customer_id'] = cid
    signals['is_fraud'] = 1 if cid in fraud_ids else 0
    all_signals.append(signals)

beh_df = pd.DataFrame(all_signals)
print(f"  Behavioral signals computed for {len(beh_df)} accounts "
      f"({beh_df['is_fraud'].sum()} fraud)")

# ─── PHASE 4: Graph Analysis ─────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 4: NetworkX Graph Analysis")
print("=" * 70)

G = build_transaction_graph(sampled_txns)
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

graph_results = []
for i, cid in enumerate(customer_list):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1} / {len(customer_list)} customers")
    ctxns = sampled_txns[sampled_txns['customer_id'] == cid]
    if len(ctxns) < 3:
        continue
    gsigs = compute_graph_signals(cid, ctxns, G)
    gsigs['customer_id'] = cid
    graph_results.append(gsigs)

graph_df = pd.DataFrame(graph_results)
print(f"  Graph signals computed for {len(graph_df)} accounts")

# ─── PHASE 5: BSI Scoring ────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 5: BSI Scoring on AMLSim Data")
print("=" * 70)

merged = beh_df.merge(graph_df, on='customer_id', how='inner')
bsi_results = []
for _, row in merged.iterrows():
    signals = row.to_dict()
    bsi = compute_bsi(signals, signals)
    bsi['customer_id'] = row['customer_id']
    bsi['is_fraud'] = row['is_fraud']
    bsi_results.append(bsi)

bsi_df = pd.DataFrame(bsi_results)
print(f"  BSI computed for {len(bsi_df)} accounts")

# ─── PHASE 6: Results & Validation ───────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 6: Cross-Validation Results")
print("=" * 70)

fraud_bsi = bsi_df[bsi_df['is_fraud'] == 1]['bsi_score']
legit_bsi = bsi_df[bsi_df['is_fraud'] == 0]['bsi_score']

print(f"\n  BSI Score Distribution:")
print(f"    Fraud accounts (n={len(fraud_bsi)}):")
print(f"      Mean: {fraud_bsi.mean():.1f}  Median: {fraud_bsi.median():.1f}  "
      f"Std: {fraud_bsi.std():.1f}")
print(f"      Min: {fraud_bsi.min():.1f}  Max: {fraud_bsi.max():.1f}")
print(f"    Legit accounts (n={len(legit_bsi)}):")
print(f"      Mean: {legit_bsi.mean():.1f}  Median: {legit_bsi.median():.1f}  "
      f"Std: {legit_bsi.std():.1f}")
print(f"      Min: {legit_bsi.min():.1f}  Max: {legit_bsi.max():.1f}")

# Separation test
mean_diff = legit_bsi.mean() - fraud_bsi.mean()
print(f"\n  Mean BSI separation (legit - fraud): {mean_diff:+.1f} points")
if mean_diff > 0:
    print("  [PASS] Legitimate accounts score HIGHER (more stable) than fraud - correct direction!")
else:
    print("  [FAIL] Unexpected: fraud accounts score higher than legitimate")

# Drift level distribution
print(f"\n  Drift Level Distribution:")
for label in ['critical', 'high', 'moderate', 'stable']:
    fraud_count = len(bsi_df[(bsi_df['is_fraud'] == 1) & (bsi_df['drift_level'] == label)])
    legit_count = len(bsi_df[(bsi_df['is_fraud'] == 0) & (bsi_df['drift_level'] == label)])
    fraud_pct = fraud_count / max(len(fraud_bsi), 1) * 100
    legit_pct = legit_count / max(len(legit_bsi), 1) * 100
    print(f"    {label:>10s}: Fraud {fraud_count:>4d} ({fraud_pct:5.1f}%)  |  "
          f"Legit {legit_count:>4d} ({legit_pct:5.1f}%)")

# Detection rate at various BSI thresholds
print(f"\n  Detection Rate at BSI Thresholds:")
for threshold in [25, 35, 50, 60, 75]:
    flagged_fraud = len(fraud_bsi[fraud_bsi <= threshold])
    flagged_legit = len(legit_bsi[legit_bsi <= threshold])
    tpr = flagged_fraud / max(len(fraud_bsi), 1) * 100
    fpr = flagged_legit / max(len(legit_bsi), 1) * 100
    print(f"    BSI <= {threshold}: TPR {tpr:5.1f}%  FPR {fpr:5.1f}%  "
          f"(caught {flagged_fraud}/{len(fraud_bsi)} fraud, "
          f"flagged {flagged_legit}/{len(legit_bsi)} legit)")

# Signal comparison
print(f"\n  Key Signal Comparison (Fraud vs Legit mean):")
signal_cols = ['amount_entropy_drift', 'burstiness_index', 'counterparty_expansion_rate',
               'structuring_score', 'benford_deviation']
for col in signal_cols:
    if col in merged.columns:
        f_mean = merged[merged['is_fraud'] == 1][col].mean()
        l_mean = merged[merged['is_fraud'] == 0][col].mean()
        direction = "^" if f_mean > l_mean else "v"
        print(f"    {col:>30s}: Fraud={f_mean:.4f}  Legit={l_mean:.4f}  {direction}")

# ─── PHASE 7: AUC-ROC ────────────────────────────────────────────────
print(f"\n  AUC-ROC Analysis:")
from sklearn.metrics import roc_auc_score, roc_curve

# Drop NaNs for clean AUC computation
clean = bsi_df.dropna(subset=['bsi_score', 'is_fraud'])
if len(clean) > 10 and clean['is_fraud'].nunique() == 2:
    risk_scores = 100 - clean['bsi_score']
    auc = roc_auc_score(clean['is_fraud'], risk_scores)
    print(f"    AUC-ROC (BSI as detector): {auc:.4f}")
    print(f"    Evaluated on: {len(clean)} accounts "
          f"({int(clean['is_fraud'].sum())} fraud, "
          f"{int((clean['is_fraud']==0).sum())} legit)")

    fpr_curve, tpr_curve, thresholds = roc_curve(clean['is_fraud'], risk_scores)
    optimal_idx = np.argmax(tpr_curve - fpr_curve)
    optimal_threshold = thresholds[optimal_idx]
    print(f"    Optimal threshold: BSI <= {100-optimal_threshold:.1f}")
    print(f"    At optimal: TPR={tpr_curve[optimal_idx]:.1%}  "
          f"FPR={fpr_curve[optimal_idx]:.1%}")
else:
    print("    Insufficient data for AUC-ROC computation")

elapsed = time.time() - start_time
print(f"\n  Total validation time: {elapsed:.1f} seconds")

# Save results
results_path = os.path.join(SCRIPT_DIR, 'amlsim_validation_results.csv')
bsi_df.to_csv(results_path, index=False)
print(f"  Results saved to {results_path}")

print("\n" + "=" * 70)
print("  CROSS-VALIDATION COMPLETE")
print("=" * 70)
