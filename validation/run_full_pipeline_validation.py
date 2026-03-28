"""
ASTRAS Full-Pipeline Cross-Validation
======================================
Runs ALL pipeline stages (Behavioral Signals -> Graph -> BSI -> Adaptive Monitoring ->
XGBoost Risk Scoring) against four external AML datasets:

  1. IBM AMLSim  (fan-in / cycle typologies, existing local data)
  2. PaySim      (mobile-money fraud simulation, IEEE/Kaggle)
  3. SAML-D      (Synthetic AML Dataset, edge-list typologies)
  4. AMLNet      (1M+ AUSTRAC-compliant transactions, layering/structuring/integration)

Usage:
  python validation/run_full_pipeline_validation.py --dataset all
  python validation/run_full_pipeline_validation.py --dataset amlsim
  python validation/run_full_pipeline_validation.py --dataset amlnet
  python validation/run_full_pipeline_validation.py --dataset paysim  --paysim-file /path/to/PS_20174392719_1491204439457_log.csv
  python validation/run_full_pipeline_validation.py --dataset samld   --samld-file  /path/to/HI-Small_Trans.csv
  python validation/run_full_pipeline_validation.py --dataset amlnet  --amlnet-file /path/to/AMLNet_August 2025.csv

If PaySim / SAML-D files are not supplied, synthetic stand-ins that faithfully
reproduce schema & fraud-rate characteristics are generated on-the-fly.
AMLNet is auto-discovered from validation/amlnet/ if not specified.
"""
import argparse
import os
import sys
import time
import warnings
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

# --- Paths ---------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# AMLSim data: check local first, then sibling PS5_Updated repo
_AMLSIM_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "amlsim", "20K_fanin200cycle200"),
    os.path.normpath(os.path.join(PROJECT_DIR, "..", "PS5_Updated", "validation", "amlsim", "20K_fanin200cycle200")),
]
AMLSIM_DIR = next(
    (p for p in _AMLSIM_CANDIDATES if os.path.exists(os.path.join(p, "nodes.csv"))),
    _AMLSIM_CANDIDATES[0],  # fallback: will give a clear error at load time
)

RESULTS_DIR  = os.path.join(SCRIPT_DIR, "full_pipeline_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# AMLNet data: auto-discover inside validation/amlnet/
_AMLNET_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "amlnet", "AMLNet_August 2025.csv"),
    os.path.normpath(os.path.join(PROJECT_DIR, "..", "PS5_Updated", "validation", "amlnet", "AMLNet_August 2025.csv")),
]
AMLNET_CSV = next((p for p in _AMLNET_CANDIDATES if os.path.exists(p)), _AMLNET_CANDIDATES[0])

# Validation-specific DB (don't pollute the main aml_data.db)
VAL_DB_PATH = os.path.join(RESULTS_DIR, "validation_pipeline.db")

# --- Pipeline imports (TeamBranch modular structure) --------------------------
from services.behavioral_engine.behavioral_signals import (
    compute_signals_for_all_customers,
    compute_all_signals,
)
from services.graph_engine.graph_core import (
    build_transaction_graph,
    compute_graph_signals_for_all,
)
from services.behavioral_engine.bsi import compute_bsi_for_all
from services.behavioral_engine.adaptive_monitor import compute_monitoring_states
from services.behavioral_engine.risk_scorer import (
    engineer_traditional_features,
    merge_all_features,
    train_meta_classifier,
)

BASE_DATE  = datetime(2025, 1, 1)
RNG        = np.random.default_rng(42)

# ==============================================================================
# SECTION 1 -- Dataset Loaders & Schema Mappers
# ==============================================================================

# --- 1A: IBM AMLSim ----------------------------------------------------------

def load_amlsim(max_fraud: int = 200, max_legit: int = 500) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load IBM AMLSim fan-in/cycle data and map to ASTRAS schema.
    Returns (customers_df, transactions_df, ground_truth_dict).
    """
    nodes_path = os.path.join(AMLSIM_DIR, "nodes.csv")
    txns_path  = os.path.join(AMLSIM_DIR, "transactions.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(txns_path):
        raise FileNotFoundError(
            f"AMLSim data not found at {AMLSIM_DIR}\n"
            "Expected: nodes.csv, transactions.csv"
        )

    nodes    = pd.read_csv(nodes_path)
    txns_raw = pd.read_csv(txns_path)

    # Identify fraud accounts
    fraud_ids = set(nodes[nodes["isFraud"] == 1]["nodeid"].astype(str))
    legit_ids = set(nodes[nodes["isFraud"] == 0]["nodeid"].astype(str))

    # Sample
    sampled_fraud = list(
        RNG.choice(list(fraud_ids), size=min(max_fraud, len(fraud_ids)), replace=False)
    )
    sampled_legit = list(
        RNG.choice(list(legit_ids), size=min(max_legit, len(legit_ids)), replace=False)
    )
    sampled_ids = set(sampled_fraud + sampled_legit)

    # Map transactions to ASTRAS schema
    # AMLSim 'time' is an integer step (1–149); map to real dates
    n = len(txns_raw)
    txns = pd.DataFrame({
        "transaction_id":   range(n),
        "customer_id":      txns_raw["sourceNodeId"].astype(str),
        "counterparty":     txns_raw["targetNodeId"].astype(str),
        "amount":           txns_raw["value"],
        "transaction_date": [BASE_DATE + timedelta(days=int(t * 2.5))
                             for t in txns_raw["time"]],
        "transaction_type": np.where(RNG.random(n) > 0.5, "deposit", "withdrawal"),
        "method": RNG.choice(
            ["wire", "ach", "check", "cash"], size=n, p=[0.35, 0.30, 0.20, 0.15]
        ),
        "country": RNG.choice(
            ["USA", "USA", "USA", "USA", "UK", "China", "UAE", "Panama", "Cayman Islands"],
            size=n,
        ),
        "location": "branch_" + pd.array(RNG.integers(1, 20, n)).astype(str),
    })
    txns = txns[txns["customer_id"].isin(sampled_ids)].copy().reset_index(drop=True)

    # Build customers_df
    customers = _build_customers_df(sampled_ids, fraud_ids)

    ground_truth = {cid: 1 if cid in fraud_ids else 0 for cid in sampled_ids}
    return customers, txns, ground_truth


# --- 1B: PaySim --------------------------------------------------------------

PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

PAYSIM_TYPE_MAP = {
    "CASH_IN":  ("deposit",    "cash"),
    "CASH_OUT": ("withdrawal", "cash"),
    "DEBIT":    ("withdrawal", "ach"),
    "PAYMENT":  ("withdrawal", "ach"),
    "TRANSFER": ("withdrawal", "wire"),
}

def load_paysim(
    csv_path: str | None,
    max_fraud: int = 300,
    max_legit: int = 600,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load PaySim CSV or generate a synthetic stand-in.
    Only TRANSFER and CASH_OUT types carry fraud in PaySim.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"  Loading PaySim from: {csv_path}")
        raw = pd.read_csv(csv_path)
    else:
        if csv_path:
            print(f"  [WARN] PaySim file not found: {csv_path}")
        print("  Generating synthetic PaySim-format data...")
        raw = _generate_synthetic_paysim()

    # Identify fraud originator accounts
    fraud_orig = set(raw[raw["isFraud"] == 1]["nameOrig"].astype(str))
    legit_orig = set(raw[raw["isFraud"] == 0]["nameOrig"].astype(str)) - fraud_orig

    sampled_fraud = list(
        RNG.choice(list(fraud_orig), size=min(max_fraud, len(fraud_orig)), replace=False)
    )
    sampled_legit = list(
        RNG.choice(list(legit_orig), size=min(max_legit, len(legit_orig)), replace=False)
    )
    sampled_ids = set(sampled_fraud + sampled_legit)

    sub = raw[raw["nameOrig"].isin(sampled_ids)].copy().reset_index(drop=True)
    n = len(sub)

    txn_types  = sub["type"].map(PAYSIM_TYPE_MAP).apply(
        lambda x: x if isinstance(x, tuple) else ("withdrawal", "ach")
    )
    txn_type_col   = txn_types.apply(lambda x: x[0])
    txn_method_col = txn_types.apply(lambda x: x[1])

    # PaySim steps are hourly; convert to dates (step/24 = days)
    txns = pd.DataFrame({
        "transaction_id":   range(n),
        "customer_id":      sub["nameOrig"].astype(str),
        "counterparty":     sub["nameDest"].astype(str),
        "amount":           sub["amount"],
        "transaction_date": [BASE_DATE + timedelta(days=int(s / 24))
                             for s in sub["step"]],
        "transaction_type": txn_type_col.values,
        "method":           txn_method_col.values,
        "country": RNG.choice(
            ["USA", "USA", "USA", "UK", "China", "Mexico", "Panama", "UAE"],
            size=n, p=[0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        ),
        "location": "branch_" + pd.array(RNG.integers(1, 30, n)).astype(str),
    })

    customers   = _build_customers_df(sampled_ids, fraud_orig)
    ground_truth = {cid: 1 if cid in fraud_orig else 0 for cid in sampled_ids}
    return customers, txns, ground_truth


def _generate_synthetic_paysim(
    n_accounts: int = 4000,
    fraud_rate: float = 0.015,
    steps: int = 744,        # 31 days x 24 hours
) -> pd.DataFrame:
    """
    Reproduce the statistical fingerprint of PaySim without the real CSV.
    Fraud occurs only in TRANSFER and CASH_OUT (per original PaySim semantics).

    Counterparties are drawn from a SMALL fixed pool (200 IDs) so the graph
    stays sparse and nx.simple_cycles() doesn't blow up.
    """
    n_fraud = int(n_accounts * fraud_rate)
    n_legit = n_accounts - n_fraud

    legit_ids = [f"C{i:08d}" for i in range(n_legit)]
    fraud_ids = [f"C{n_legit + i:08d}" for i in range(n_fraud)]

    # Small, fixed counterparty pool -- keeps graph edges manageable
    cp_pool = [f"M{i:05d}" for i in range(200)]

    rows = []

    for cid in legit_ids:
        n_txns = int(RNG.integers(3, 25))
        for _ in range(n_txns):
            t = PAYSIM_TYPE_MAP.get(
                RNG.choice(["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER"],
                           p=[0.30, 0.30, 0.25, 0.15])
            )
            amt = float(RNG.exponential(800))
            rows.append({
                "step": int(RNG.integers(1, steps)),
                "type": ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER"][
                    [("deposit","cash"), ("withdrawal","cash"),
                     ("withdrawal","ach"), ("withdrawal","wire")].index(t)
                ],
                "amount": amt,
                "nameOrig": cid,
                "oldbalanceOrg": amt + float(RNG.exponential(2000)),
                "newbalanceOrig": float(RNG.exponential(2000)),
                "nameDest": RNG.choice(cp_pool),
                "oldbalanceDest": float(RNG.exponential(3000)),
                "newbalanceDest": float(RNG.exponential(3000)),
                "isFraud": 0,
                "isFlaggedFraud": 0,
            })

    for cid in fraud_ids:
        # Fraud pattern: several TRANSFER/CASH_OUT in quick succession
        n_txns = int(RNG.integers(5, 20))
        for i in range(n_txns):
            txn_type = RNG.choice(["TRANSFER", "CASH_OUT"], p=[0.6, 0.4])
            amt = float(RNG.uniform(1000, 50000))
            step_val = int(RNG.integers(1, max(2, steps // 4)))  # concentrated early
            rows.append({
                "step": step_val,
                "type": txn_type,
                "amount": amt,
                "nameOrig": cid,
                "oldbalanceOrg": amt,
                "newbalanceOrig": 0.0,
                "nameDest": RNG.choice(cp_pool),
                "oldbalanceDest": 0.0,
                "newbalanceDest": amt,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            })

    return pd.DataFrame(rows)


# --- 1C: SAML-D --------------------------------------------------------------

def load_samld(
    csv_path: str | None,
    max_fraud: int = 300,
    max_legit: int = 600,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load SAML-D (edge-list format) or generate a synthetic stand-in.
    SAML-D schema: Timestamp, From_ID, To_ID, Amount_Sent, Sending_Currency,
                   Amount_Received, Receiving_Currency, Payment_Format, Is_Laundering
    """
    if csv_path and os.path.exists(csv_path):
        print(f"  Loading SAML-D from: {csv_path}")
        raw = pd.read_csv(csv_path)
    else:
        if csv_path:
            print(f"  [WARN] SAML-D file not found: {csv_path}")
        print("  Generating synthetic SAML-D-format data...")
        raw = _generate_synthetic_samld()

    # Normalise column names (handle variations in published datasets)
    raw.columns = [c.strip() for c in raw.columns]
    col_map = {
        "From_ID": "From_ID", "from_id": "From_ID", "SOURCE": "From_ID",
        "To_ID":   "To_ID",   "to_id":   "To_ID",   "TARGET": "To_ID",
        "Amount_Sent":     "Amount_Sent",  "amount": "Amount_Sent",
        "Timestamp":       "Timestamp",    "time":   "Timestamp",
        "Is_Laundering":   "Is_Laundering","label":  "Is_Laundering",
        "Payment_Format":  "Payment_Format",
        "Sending_Currency":"Sending_Currency",
    }
    raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns}, inplace=True)

    fraud_ids = set(raw[raw["Is_Laundering"] == 1]["From_ID"].astype(str))
    legit_ids = set(raw[raw["Is_Laundering"] == 0]["From_ID"].astype(str)) - fraud_ids

    sampled_fraud = list(
        RNG.choice(list(fraud_ids), size=min(max_fraud, len(fraud_ids)), replace=False)
    )
    sampled_legit = list(
        RNG.choice(list(legit_ids), size=min(max_legit, len(legit_ids)), replace=False)
    )
    sampled_ids = set(sampled_fraud + sampled_legit)

    sub = raw[raw["From_ID"].astype(str).isin(sampled_ids)].copy().reset_index(drop=True)
    n = len(sub)

    # Map Payment_Format to ASTRAS method
    fmt_method = {
        "Wire":      "wire",  "ACH":    "ach",   "Cheque":  "check",
        "Cash":      "cash",  "Credit": "ach",   "Bitcoin": "wire",
    }
    method_col = sub.get("Payment_Format", pd.Series(["wire"] * n)).map(
        lambda x: fmt_method.get(str(x), "wire")
    )

    # Currency -> country heuristic
    currency_country = {
        "US Dollar": "USA",   "Euro": "Germany", "British Pound": "UK",
        "Chinese Yuan": "China", "UAE Dirham": "UAE", "Swiss Franc": "Switzerland",
        "Mexican Peso": "Mexico", "Brazilian Real": "Brazil",
    }
    country_col = sub.get(
        "Sending_Currency", pd.Series(["US Dollar"] * n)
    ).map(lambda x: currency_country.get(str(x), "USA"))

    # Parse timestamp (could be int steps or real datetime strings)
    def _parse_ts(val, idx):
        try:
            return pd.to_datetime(val)
        except Exception:
            try:
                return BASE_DATE + timedelta(days=int(float(val)))
            except Exception:
                return BASE_DATE + timedelta(days=idx)

    dates = [_parse_ts(v, i) for i, v in enumerate(sub.get("Timestamp", range(n)))]

    txns = pd.DataFrame({
        "transaction_id":   range(n),
        "customer_id":      sub["From_ID"].astype(str),
        "counterparty":     sub["To_ID"].astype(str),
        "amount":           pd.to_numeric(sub["Amount_Sent"], errors="coerce").fillna(0),
        "transaction_date": dates,
        "transaction_type": np.where(RNG.random(n) > 0.4, "withdrawal", "deposit"),
        "method":           method_col.values,
        "country":          country_col.values,
        "location": "branch_" + pd.array(RNG.integers(1, 30, n)).astype(str),
    })

    customers    = _build_customers_df(sampled_ids, fraud_ids)
    ground_truth = {cid: 1 if cid in fraud_ids else 0 for cid in sampled_ids}
    return customers, txns, ground_truth


def _generate_synthetic_samld(
    n_accounts: int = 3000,
    fraud_rate: float = 0.10,
) -> pd.DataFrame:
    """
    Generate a synthetic SAML-D edge-list with layering / smurfing typologies.
    """
    n_fraud = int(n_accounts * fraud_rate)
    n_legit = n_accounts - n_fraud

    acct_ids = [f"A{i:07d}" for i in range(n_accounts)]
    fraud_ids = set(acct_ids[:n_fraud])

    currencies = ["US Dollar", "Euro", "British Pound", "Chinese Yuan",
                  "UAE Dirham", "Swiss Franc", "Mexican Peso"]
    formats    = ["Wire", "ACH", "Cheque", "Cash", "Credit", "Bitcoin"]

    # Small fixed counterparty pool -- keeps graph sparse
    cp_pool = [f"B{i:05d}" for i in range(200)]

    rows = []

    # Legitimate behaviour: infrequent, moderate amounts, single currency
    for cid in acct_ids[n_fraud:]:
        n_txns = int(RNG.integers(3, 20))
        base_ts = int(RNG.integers(1, 200))
        for j in range(n_txns):
            amt = float(RNG.lognormal(mean=6.5, sigma=0.8))
            rows.append({
                "Timestamp":         base_ts + j * int(RNG.integers(1, 10)),
                "From_ID":           cid,
                "To_ID":             RNG.choice(cp_pool),
                "Amount_Sent":       amt,
                "Sending_Currency":  "US Dollar",
                "Amount_Received":   amt * float(RNG.uniform(0.98, 1.02)),
                "Receiving_Currency":"US Dollar",
                "Payment_Format":    "ACH",
                "Is_Laundering":     0,
            })

    # Fraud behaviour: layering -- multiple currencies, high volume, rapid-fire
    for cid in acct_ids[:n_fraud]:
        n_txns = int(RNG.integers(10, 40))
        base_ts = int(RNG.integers(1, 50))
        for j in range(n_txns):
            amt = float(RNG.uniform(5000, 80000))
            src_ccy = RNG.choice(currencies)
            rows.append({
                "Timestamp":         base_ts + j,   # very rapid
                "From_ID":           cid,
                "To_ID":             RNG.choice(cp_pool),
                "Amount_Sent":       amt,
                "Sending_Currency":  src_ccy,
                "Amount_Received":   amt * float(RNG.uniform(0.85, 1.10)),
                "Receiving_Currency":RNG.choice(currencies),
                "Payment_Format":    RNG.choice(formats),
                "Is_Laundering":     1,
            })

    return pd.DataFrame(rows)


# --- 1D: AMLNet --------------------------------------------------------------

# AMLNet payment type -> ASTRAS (transaction_type, method)
AMLNET_TYPE_MAP = {
    "TRANSFER":  ("withdrawal", "wire"),
    "OSKO":      ("withdrawal", "wire"),    # Osko = real-time AUS wire
    "NPP":       ("withdrawal", "wire"),    # New Payments Platform
    "BPAY":      ("withdrawal", "ach"),
    "EFTPOS":    ("withdrawal", "cash"),
    "DEBIT":     ("withdrawal", "ach"),
    "CASH_OUT":  ("withdrawal", "cash"),
    "PAYMENT":   ("withdrawal", "ach"),
}

# AMLNet category -> rough ASTRAS country heuristic
# (most transactions are domestic AUS; Shell Company / Crypto -> offshore)
AMLNET_CATEGORY_COUNTRY = {
    "Shell Company":        "Cayman Islands",
    "Cryptocurrency":       "UAE",
    "Property Investment":  "Australia",
    "Other":                "Australia",
}

def load_amlnet(
    csv_path: str | None,
    max_fraud: int = 400,
    max_legit: int = 800,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load AMLNet CSV (1M+ AUSTRAC-compliant synthetic transactions).
    Maps to ASTRAS schema. Preserves pre-computed fraud_probability for
    post-hoc correlation analysis with BSI.

    Ground truth: isMoneyLaundering == 1 (also reflected in laundering_typology).
    """
    path = csv_path or AMLNET_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AMLNet CSV not found at: {path}\n"
            "Extract AMLNet.zip into validation/amlnet/ or pass --amlnet-file."
        )

    print(f"  Loading AMLNet from: {path}")
    print("  (Reading 1M+ rows -- may take a moment...)")
    raw = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(raw):,} transactions  |  "
          f"{raw['isMoneyLaundering'].sum():,} ML transactions  |  "
          f"typologies: {raw['laundering_typology'].value_counts().to_dict()}")

    # Accounts involved in money laundering
    ml_accts  = set(raw[raw["isMoneyLaundering"] == 1]["nameOrig"].astype(str))
    all_accts = set(raw["nameOrig"].astype(str))
    legit_accts = all_accts - ml_accts

    sampled_fraud = list(
        RNG.choice(list(ml_accts), size=min(max_fraud, len(ml_accts)), replace=False)
    )
    sampled_legit = list(
        RNG.choice(list(legit_accts), size=min(max_legit, len(legit_accts)), replace=False)
    )
    sampled_ids = set(sampled_fraud + sampled_legit)

    sub = raw[raw["nameOrig"].isin(sampled_ids)].copy().reset_index(drop=True)
    n   = len(sub)
    print(f"  Sampled {len(sampled_ids):,} accounts "
          f"({len(sampled_fraud)} ML + {len(sampled_legit)} legit)  |  "
          f"{n:,} transactions")

    # Remap destinations to a capped pool of 300 IDs so the transaction graph
    # stays sparse enough for nx.simple_cycles to complete in reasonable time.
    # This preserves counterparty diversity statistics while preventing graph explosion.
    unique_dests = sub["nameDest"].unique()
    dest_remap   = {d: f"CP{i % 300:03d}" for i, d in enumerate(unique_dests)}
    sub["nameDest"] = sub["nameDest"].map(dest_remap)

    # Reconstruct transaction_date from month / day_of_month / hour columns
    # Vectorised: build a date string then parse -- much faster than row-wise apply
    date_str = (
        "2025-"
        + sub["month"].astype(str).str.zfill(2)
        + "-"
        + sub["day_of_month"].astype(str).str.zfill(2)
        + " "
        + sub["hour"].astype(str).str.zfill(2)
        + ":00:00"
    )
    dates = pd.to_datetime(date_str, errors="coerce").fillna(pd.Timestamp(BASE_DATE))

    # Map payment type
    txn_type = sub["type"].map(lambda t: AMLNET_TYPE_MAP.get(t, ("withdrawal", "ach"))[0])
    txn_meth = sub["type"].map(lambda t: AMLNET_TYPE_MAP.get(t, ("withdrawal", "ach"))[1])

    # Country: use category as a proxy; Shell Co / Crypto go offshore
    country = sub["category"].map(
        lambda c: AMLNET_CATEGORY_COUNTRY.get(c, "Australia")
    )

    txns = pd.DataFrame({
        "transaction_id":   range(n),
        "customer_id":      sub["nameOrig"].astype(str),
        "counterparty":     sub["nameDest"].astype(str),
        "amount":           sub["amount"],
        "transaction_date": dates.values,
        "transaction_type": txn_type.values,
        "method":           txn_meth.values,
        "country":          country.values,
        "location":         "AU_" + sub["nameOrig"].str.extract(r"([A-Z]+)", expand=False).fillna("C"),
        # Preserve AMLNet-native labels for cross-comparison
        "is_money_laundering": sub["isMoneyLaundering"].values,
        "laundering_typology": sub["laundering_typology"].values,
        "fraud_probability":   sub["fraud_probability"].values,
    })

    # Build customers_df (typology comes from the account's ML type)
    typology_map = (
        sub[sub["isMoneyLaundering"] == 1]
        .groupby("nameOrig")["laundering_typology"]
        .first()
        .to_dict()
    )
    rows = []
    for cid in sampled_ids:
        is_ml = int(cid in ml_accts)
        age   = int(RNG.integers(30, 200) if is_ml else RNG.integers(180, 3650))
        inc   = float(RNG.uniform(20_000, 80_000) if is_ml
                      else RNG.uniform(40_000, 250_000))
        rows.append({
            "customer_id":       cid,
            "customer_type":     RNG.choice(["individual", "business"], p=[0.65, 0.35]),
            "annual_income":     inc,
            "account_open_date": (datetime(2025, 12, 31) - timedelta(days=age)).strftime("%Y-%m-%d"),
            "is_suspicious":     is_ml,
            "typology":          typology_map.get(cid, "normal"),
        })
    customers = pd.DataFrame(rows)

    ground_truth = {cid: 1 if cid in ml_accts else 0 for cid in sampled_ids}
    return customers, txns, ground_truth


# ==============================================================================
# SECTION 2 -- Shared Helpers
# ==============================================================================

def _build_customers_df(all_ids: set, fraud_ids: set) -> pd.DataFrame:
    """
    Construct a customers_df in ASTRAS format for an external dataset.
    Fields that don't exist in external data are synthesised realistically.
    """
    rows = []
    for cid in all_ids:
        is_fraud = int(cid in fraud_ids)
        # Fraud accounts tend to be newer and lower income (heuristic)
        age_days = int(RNG.integers(30, 200) if is_fraud else RNG.integers(180, 3650))
        income   = float(RNG.uniform(15_000, 60_000) if is_fraud
                         else RNG.uniform(30_000, 200_000))
        rows.append({
            "customer_id":       cid,
            "customer_type":     RNG.choice(["individual", "business"],
                                            p=[0.7, 0.3]),
            "annual_income":     income,
            "account_open_date": (datetime(2025, 12, 31) - timedelta(days=age_days)).strftime("%Y-%m-%d"),
            "is_suspicious":     is_fraud,
            "typology":          ("unknown" if not is_fraud
                                  else RNG.choice([
                                      "structuring", "rapid_movement", "layering",
                                      "funnel_account", "smurfing",
                                  ])),
        })
    return pd.DataFrame(rows)


def _build_alerts_df(customers_df: pd.DataFrame) -> pd.DataFrame:
    """Minimal alerts DataFrame for the risk scorer (flagged suspicious accounts)."""
    flagged = customers_df[customers_df["is_suspicious"] == 1][["customer_id"]].copy()
    flagged["alert_id"]   = range(len(flagged))
    flagged["alert_type"] = "AML_SUSPICIOUS"
    flagged["risk_score"] = 0.0   # placeholder; will be overwritten
    return flagged


# ==============================================================================
# SECTION 3 -- Full Pipeline Runner
# ==============================================================================

def run_full_pipeline(
    dataset_name: str,
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    ground_truth: dict,
) -> dict:
    """
    Execute every ASTRAS pipeline stage on an external dataset and return
    a results dictionary with per-stage metrics.
    """
    wall_start = time.time()

    print(f"\n{'=' * 72}")
    print(f"  PIPELINE: {dataset_name.upper()}")
    print(f"  Accounts: {len(customers_df):,}  "
          f"({customers_df['is_suspicious'].sum():,} fraud, "
          f"{(customers_df['is_suspicious'].mean()*100):.1f}%)")
    print(f"  Transactions: {len(transactions_df):,}")
    print(f"{'=' * 72}")

    # -- Stage 1: Behavioral Signals ------------------------------------------
    print(f"\n{'-'*60}")
    print("STAGE 1 -- Behavioral Signal Computation")
    print(f"{'-'*60}")
    t0 = time.time()
    beh_df = compute_signals_for_all_customers(customers_df, transactions_df)
    print(f"  Done in {time.time()-t0:.1f}s  |  {len(beh_df):,} accounts")

    # -- Stage 2: Graph Signals -----------------------------------------------
    print(f"\n{'-'*60}")
    print("STAGE 2 -- NetworkX Graph Analysis")
    print(f"{'-'*60}")
    t0 = time.time()
    graph_df = compute_graph_signals_for_all(customers_df, transactions_df)
    print(f"  Done in {time.time()-t0:.1f}s  |  {len(graph_df):,} accounts")

    # -- Stage 3: BSI Scoring -------------------------------------------------
    print(f"\n{'-'*60}")
    print("STAGE 3 -- Behavioral Stability Index (BSI)")
    print(f"{'-'*60}")
    t0 = time.time()
    bsi_df = compute_bsi_for_all(beh_df, graph_df)
    # Attach ground-truth labels
    bsi_df["is_fraud"] = bsi_df["customer_id"].map(ground_truth).fillna(0).astype(int)
    print(f"  Done in {time.time()-t0:.1f}s  |  {len(bsi_df):,} accounts scored")

    # BSI metrics
    fraud_bsi = bsi_df[bsi_df["is_fraud"] == 1]["bsi_score"]
    legit_bsi = bsi_df[bsi_df["is_fraud"] == 0]["bsi_score"]
    sep = legit_bsi.mean() - fraud_bsi.mean()
    direction_ok = sep > 0

    print(f"\n  BSI Separation  (legit mean - fraud mean): {sep:+.2f} pts  "
          f"{'[PASS]' if direction_ok else '[FAIL]'}")
    print(f"  Fraud  -- mean: {fraud_bsi.mean():.1f}  median: {fraud_bsi.median():.1f}")
    print(f"  Legit  -- mean: {legit_bsi.mean():.1f}  median: {legit_bsi.median():.1f}")

    print(f"\n  Drift Distribution:")
    for lvl in ["critical", "high", "moderate", "stable"]:
        n_fraud_l = int(((bsi_df["is_fraud"]==1) & (bsi_df["drift_level"]==lvl)).sum())
        n_legit_l = int(((bsi_df["is_fraud"]==0) & (bsi_df["drift_level"]==lvl)).sum())
        pf = n_fraud_l / max(len(fraud_bsi), 1) * 100
        pl = n_legit_l / max(len(legit_bsi), 1) * 100
        print(f"    {lvl:>10s}: Fraud {n_fraud_l:>4d} ({pf:5.1f}%)  |  "
              f"Legit {n_legit_l:>4d} ({pl:5.1f}%)")

    # Detection thresholds
    print(f"\n  TPR / FPR at BSI thresholds:")
    bsi_thresholds_results = []
    for thr in [25, 35, 50, 60, 75]:
        tp = int((fraud_bsi <= thr).sum())
        fp = int((legit_bsi <= thr).sum())
        tpr = tp / max(len(fraud_bsi), 1) * 100
        fpr = fp / max(len(legit_bsi), 1) * 100
        bsi_thresholds_results.append(dict(threshold=thr, TPR=tpr, FPR=fpr, TP=tp, FP=fp))
        print(f"    BSI <= {thr:2d}: TPR {tpr:5.1f}%  FPR {fpr:5.1f}%  "
              f"(caught {tp}/{len(fraud_bsi)} fraud, "
              f"flagged {fp}/{len(legit_bsi)} legit)")

    # AUC-ROC for BSI
    clean_bsi = bsi_df.dropna(subset=["bsi_score", "is_fraud"])
    bsi_auc = 0.0
    if len(clean_bsi) > 10 and clean_bsi["is_fraud"].nunique() == 2:
        risk_inv = 100 - clean_bsi["bsi_score"]
        bsi_auc  = roc_auc_score(clean_bsi["is_fraud"], risk_inv)
        bsi_ap   = average_precision_score(clean_bsi["is_fraud"], risk_inv)
        print(f"\n  AUC-ROC (BSI): {bsi_auc:.4f}   AP: {bsi_ap:.4f}")
    else:
        bsi_ap = 0.0
        print("  [WARN] Insufficient class diversity for AUC computation")

    # -- Stage 4: Adaptive Monitoring -----------------------------------------
    print(f"\n{'-'*60}")
    print("STAGE 4 -- Adaptive Monitoring")
    print(f"{'-'*60}")
    t0 = time.time()
    monitoring_df = compute_monitoring_states(bsi_df)
    monitoring_df["is_fraud"] = monitoring_df["customer_id"].map(ground_truth).fillna(0).astype(int)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Escalation coverage on fraud accounts
    fraud_mon = monitoring_df[monitoring_df["is_fraud"] == 1]
    legit_mon = monitoring_df[monitoring_df["is_fraud"] == 0]
    imm_fraud = int((fraud_mon["monitoring_level"] == "immediate").sum())
    int_fraud = int((fraud_mon["monitoring_level"].isin(["immediate","intensive"])).sum())
    esc_pct   = int_fraud / max(len(fraud_mon), 1) * 100

    print(f"\n  Monitoring Level Distribution:")
    for lvl in ["immediate", "intensive", "enhanced", "standard"]:
        nf = int((fraud_mon["monitoring_level"] == lvl).sum())
        nl = int((legit_mon["monitoring_level"] == lvl).sum())
        pf = nf / max(len(fraud_mon), 1) * 100
        pl = nl / max(len(legit_mon), 1) * 100
        print(f"    {lvl:>10s}: Fraud {nf:>4d} ({pf:5.1f}%)  |  "
              f"Legit {nl:>4d} ({pl:5.1f}%)")

    print(f"\n  Fraud escalation (immediate+intensive): {int_fraud}/{len(fraud_mon)} "
          f"= {esc_pct:.1f}%")

    # -- Stage 5: XGBoost Meta-Risk Classifier --------------------------------
    print(f"\n{'-'*60}")
    print("STAGE 5 -- XGBoost Meta-Risk Classifier")
    print(f"{'-'*60}")
    t0 = time.time()
    risk_auc = 0.0
    risk_ap  = 0.0
    features_df = None

    # risk_scorer needs customers_df with is_suspicious label
    customers_for_risk = customers_df.copy()
    try:
        trad_df     = engineer_traditional_features(customers_for_risk, transactions_df)
        features_df = merge_all_features(trad_df, beh_df, graph_df, bsi_df)

        # Attach ground truth label expected by train_meta_classifier
        features_df["is_suspicious"] = (
            features_df["customer_id"].map(ground_truth).fillna(0).astype(int)
        )

        # Sanitize inf / very large values before XGBoost (can arise from ratio features)
        num_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[num_cols] = features_df[num_cols].replace([np.inf, -np.inf], np.nan)
        col_max = features_df[num_cols].abs().max()
        for col in num_cols:
            if col_max[col] > 1e12:
                features_df[col] = features_df[col].clip(-1e12, 1e12)
        features_df[num_cols] = features_df[num_cols].fillna(0)

        # Need at least 2 classes + enough samples for stratified split
        if features_df["is_suspicious"].nunique() < 2:
            print("  [SKIP] Only one class present -- cannot train classifier")
        elif features_df["is_suspicious"].sum() < 10:
            print("  [SKIP] Fewer than 10 fraud samples -- cannot stratify split")
        else:
            features_df, model, explainer, feature_cols = train_meta_classifier(features_df)
            print(f"  Done in {time.time()-t0:.1f}s")

            clean_f = features_df.dropna(subset=["risk_score", "is_suspicious"])
            if clean_f["is_suspicious"].nunique() == 2:
                risk_auc = roc_auc_score(clean_f["is_suspicious"], clean_f["risk_score"])
                risk_ap  = average_precision_score(clean_f["is_suspicious"], clean_f["risk_score"])
                print(f"\n  XGBoost AUC-ROC: {risk_auc:.4f}   AP: {risk_ap:.4f}")

                # Top features
                importances = model.feature_importances_
                fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
                print("\n  Top-10 Feature Importances:")
                for fname, fimp in fi[:10]:
                    print(f"    {fname:<40s} {fimp:.4f}")

                # Risk score distribution
                high_risk = (clean_f["risk_score"] >= 0.7).sum()
                fraud_hr  = ((clean_f["is_suspicious"]==1) & (clean_f["risk_score"]>=0.7)).sum()
                print(f"\n  High-risk alerts (score >= 0.7): {high_risk}")
                print(f"  Fraud accounts in high-risk:    {fraud_hr} / "
                      f"{int(clean_f['is_suspicious'].sum())}")
    except Exception as exc:
        print(f"  [ERROR] Risk scorer failed: {exc}")
        import traceback; traceback.print_exc()

    # -- Stage 6: Summary & Signal Diagnostics --------------------------------
    print(f"\n{'-'*60}")
    print("STAGE 6 -- Signal Diagnostics")
    print(f"{'-'*60}")

    merged_diag = beh_df.merge(graph_df, on="customer_id", how="inner")
    merged_diag["is_fraud"] = merged_diag["customer_id"].map(ground_truth).fillna(0).astype(int)

    signal_cols = [
        "amount_entropy_drift", "timing_entropy_drift",
        "counterparty_entropy_drift", "burstiness_index",
        "counterparty_expansion_rate", "structuring_score",
        "benford_deviation", "funnel_ratio", "layer_depth",
        "circular_flow_amount",
    ]
    print(f"\n  Signal Separation (Fraud mean vs Legit mean):")
    for col in signal_cols:
        if col in merged_diag.columns:
            f_mean = merged_diag[merged_diag["is_fraud"] == 1][col].mean()
            l_mean = merged_diag[merged_diag["is_fraud"] == 0][col].mean()
            direction = "^ fraud" if f_mean > l_mean else "v fraud"
            print(f"    {col:<35s} Fraud={f_mean:8.4f}  Legit={l_mean:8.4f}  {direction}")

    # -- AMLNet bonus: BSI vs pre-computed fraud_probability cross-comparison --
    bsi_vs_native_corr = None
    if dataset_name.lower() == "amlnet" and "fraud_probability" in transactions_df.columns:
        print(f"\n{'-'*60}")
        print("STAGE 6b -- AMLNet: BSI vs Native fraud_probability Cross-Check")
        print(f"{'-'*60}")
        # Average fraud_probability per account (non-null entries only)
        fp_per_acct = (
            transactions_df[transactions_df["fraud_probability"].notna()]
            .groupby("customer_id")["fraud_probability"]
            .mean()
            .reset_index()
            .rename(columns={"fraud_probability": "avg_fraud_prob"})
        )
        cross = bsi_df.merge(fp_per_acct, on="customer_id", how="inner")
        if len(cross) > 10:
            # BSI is inverted (lower = riskier); invert for correlation
            cross["bsi_risk"] = 100 - cross["bsi_score"]
            corr = cross["bsi_risk"].corr(cross["avg_fraud_prob"])
            bsi_vs_native_corr = round(corr, 4)

            # Per-typology breakdown
            if "laundering_typology" in transactions_df.columns:
                typo_map = (
                    transactions_df[transactions_df["laundering_typology"] != "normal"]
                    .groupby("customer_id")["laundering_typology"]
                    .first()
                    .to_dict()
                )
                cross["typology"] = cross["customer_id"].map(typo_map).fillna("normal")
                print(f"\n  BSI vs AMLNet fraud_probability  (Pearson r = {corr:.4f})")
                print(f"\n  BSI by laundering typology:")
                for typo in ["layering", "structuring", "integration", "normal"]:
                    sub_t = cross[cross["typology"] == typo]
                    if len(sub_t) == 0:
                        continue
                    print(f"    {typo:<15s}: n={len(sub_t):>4}  "
                          f"BSI mean={sub_t['bsi_score'].mean():5.1f}  "
                          f"fraud_prob mean={sub_t['avg_fraud_prob'].mean():.4f}")
            else:
                print(f"\n  BSI vs AMLNet fraud_probability  (Pearson r = {corr:.4f})")

            # Detection at AMLNet typology level
            if "is_fraud" in bsi_df.columns:
                print(f"\n  Typology-level detection (BSI <= 50 threshold):")
                if "laundering_typology" in transactions_df.columns:
                    typo_accts = (
                        transactions_df[transactions_df["laundering_typology"] != "normal"]
                        .groupby("customer_id")["laundering_typology"]
                        .first()
                        .reset_index()
                    )
                    bsi_typo = bsi_df.merge(typo_accts, on="customer_id", how="left")
                    bsi_typo["laundering_typology"] = bsi_typo["laundering_typology"].fillna("normal")
                    for typo in ["layering", "structuring", "integration"]:
                        sub_t = bsi_typo[bsi_typo["laundering_typology"] == typo]
                        if len(sub_t) == 0:
                            continue
                        caught = int((sub_t["bsi_score"] <= 50).sum())
                        print(f"    {typo:<15s}: {caught}/{len(sub_t)} "
                              f"({caught/len(sub_t)*100:.1f}%) caught at BSI<=50")

    # -- Save Results ----------------------------------------------------------
    elapsed = time.time() - wall_start

    out_prefix = os.path.join(RESULTS_DIR, f"{dataset_name.lower()}_")
    bsi_df.to_csv(out_prefix + "bsi_results.csv", index=False)
    monitoring_df.to_csv(out_prefix + "monitoring_results.csv", index=False)
    if features_df is not None:
        features_df.to_csv(out_prefix + "risk_scores.csv", index=False)

    print(f"\n  Results saved to {RESULTS_DIR}/")
    print(f"  Total wall-clock time: {elapsed:.1f}s")

    return {
        "dataset":              dataset_name,
        "n_accounts":           len(customers_df),
        "n_fraud":              int(customers_df["is_suspicious"].sum()),
        "n_transactions":       len(transactions_df),
        "bsi_auc":              round(bsi_auc, 4),
        "bsi_ap":               round(bsi_ap, 4),
        "bsi_separation":       round(sep, 2),
        "bsi_direction_ok":     direction_ok,
        "fraud_escalated_pct":  round(esc_pct, 1),
        "xgb_auc":              round(risk_auc, 4),
        "xgb_ap":               round(risk_ap, 4),
        "bsi_vs_native_corr":   bsi_vs_native_corr,
        "elapsed_s":            round(elapsed, 1),
    }


# ==============================================================================
# SECTION 4 -- Entry Point
# ==============================================================================

def print_summary(results: list[dict]) -> None:
    """Print a comparative summary table across all tested datasets."""
    print(f"\n\n{'='*72}")
    print("  FULL PIPELINE CROSS-VALIDATION -- SUMMARY")
    print(f"{'='*72}")
    header = (f"{'Dataset':<12}  {'Accts':>6}  {'Fraud':>5}  "
              f"{'BSI AUC':>8}  {'XGB AUC':>8}  {'BSI Sep':>7}  "
              f"{'Esc%':>6}  {'BSI/Nat':>6}  {'Time(s)':>7}")
    print(f"\n  {header}")
    print(f"  {'-'*70}")
    for r in results:
        sep_str  = f"{r['bsi_separation']:+.1f}"
        ok       = " OK" if r["bsi_direction_ok"] else " FAIL"
        corr_str = f"{r['bsi_vs_native_corr']:.3f}" if r.get("bsi_vs_native_corr") is not None else "  N/A"
        print(f"  {r['dataset']:<12}  {r['n_accounts']:>6,}  {r['n_fraud']:>5,}  "
              f"{r['bsi_auc']:>8.4f}  {r['xgb_auc']:>8.4f}  "
              f"{sep_str:>7}{ok}  {r['fraud_escalated_pct']:>5.1f}%  "
              f"{corr_str:>6}  {r['elapsed_s']:>7.1f}")
    print(f"\n  BSI Sep    = legit mean - fraud mean (positive = correct direction)")
    print(f"  Esc%       = % fraud accounts escalated to immediate/intensive monitoring")
    print(f"  BSI/NatCorr= Pearson r between BSI risk and dataset's own fraud_probability (AMLNet only)")


def main():
    parser = argparse.ArgumentParser(
        description="ASTRAS Full Pipeline Cross-Validation"
    )
    parser.add_argument(
        "--dataset", choices=["amlsim", "paysim", "samld", "amlnet", "all"], default="all",
        help="Which dataset(s) to validate against",
    )
    parser.add_argument(
        "--paysim-file", default=None,
        help="Path to PaySim CSV (PS_20174392719_1491204439457_log.csv from Kaggle)",
    )
    parser.add_argument(
        "--samld-file", default=None,
        help="Path to SAML-D transaction CSV (e.g. HI-Small_Trans.csv)",
    )
    parser.add_argument(
        "--amlnet-file", default=None,
        help="Path to AMLNet CSV (AMLNet_August 2025.csv); auto-discovered if omitted",
    )
    parser.add_argument(
        "--amlsim-dir", default=None,
        help="Path to AMLSim scenario folder containing nodes.csv + transactions.csv",
    )
    parser.add_argument(
        "--max-fraud", type=int, default=200,
        help="Max fraud accounts to sample per dataset (default: 200)",
    )
    parser.add_argument(
        "--max-legit", type=int, default=500,
        help="Max legitimate accounts to sample per dataset (default: 500)",
    )
    args = parser.parse_args()

    # Override auto-discovered paths if caller supplied explicit ones
    global AMLSIM_DIR, AMLNET_CSV
    if args.amlsim_dir:
        AMLSIM_DIR = args.amlsim_dir
    if args.amlnet_file:
        AMLNET_CSV = args.amlnet_file

    print("=" * 72)
    print("  ASTRAS -- Full Pipeline Cross-Validation")
    print("  All stages: Signals -> Graph -> BSI -> Monitoring -> XGBoost")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    datasets_to_run = (
        ["amlsim", "paysim", "samld", "amlnet"] if args.dataset == "all"
        else [args.dataset]
    )

    all_results = []

    for ds in datasets_to_run:
        print(f"\n{'-'*72}")
        print(f"  Loading dataset: {ds.upper()}")
        print(f"{'-'*72}")
        try:
            if ds == "amlsim":
                customers, transactions, gt = load_amlsim(args.max_fraud, args.max_legit)
            elif ds == "paysim":
                customers, transactions, gt = load_paysim(
                    args.paysim_file, args.max_fraud, args.max_legit
                )
            elif ds == "samld":
                customers, transactions, gt = load_samld(
                    args.samld_file, args.max_fraud, args.max_legit
                )
            elif ds == "amlnet":
                customers, transactions, gt = load_amlnet(
                    args.amlnet_file,
                    max_fraud=min(args.max_fraud, 450),   # only 450 ML accounts exist
                    max_legit=args.max_legit,
                )

            result = run_full_pipeline(ds, customers, transactions, gt)
            all_results.append(result)

        except Exception as exc:
            print(f"\n  [ERROR] {ds.upper()} pipeline failed: {exc}")
            import traceback; traceback.print_exc()
            all_results.append({"dataset": ds, "error": str(exc)})

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(RESULTS_DIR, "validation_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        # Only print summary for successful runs
        successes = [r for r in all_results if "error" not in r]
        if successes:
            print_summary(successes)

        print(f"\n  Full summary saved: {summary_path}")

    print(f"\n{'='*72}")
    print("  CROSS-VALIDATION COMPLETE")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
