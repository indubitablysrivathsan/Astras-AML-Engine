"""
XGBoost Meta-Risk Classifier
Operates as a meta-risk fusion layer over behavioral signals,
producing composite risk scores with full SHAP explainability.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import joblib
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import XGBOOST_PARAMS, MODELS_DIR, DB_PATH, HIGH_RISK_COUNTRIES, COUNTRY_PPL, DEFAULT_PPL


def engineer_traditional_features(customers_df, transactions_df):
    """Compute traditional transaction-level features."""
    print("Engineering traditional features...")
    features_list = []

    for idx, customer in customers_df.iterrows():
        ctxns = transactions_df[transactions_df['customer_id'] == customer['customer_id']].copy()
        if len(ctxns) == 0:
            continue

        ctxns['transaction_date'] = pd.to_datetime(ctxns['transaction_date'])
        date_range = max((ctxns['transaction_date'].max() - ctxns['transaction_date'].min()).days, 1)

        deposits = ctxns[ctxns['transaction_type'] == 'deposit']
        withdrawals = ctxns[ctxns['transaction_type'] == 'withdrawal']
        cash = ctxns[ctxns['method'] == 'cash']
        wire = ctxns[ctxns['method'] == 'wire']
        intl = ctxns[ctxns['country'] != 'USA']
        high_risk = ctxns[ctxns['country'].isin(HIGH_RISK_COUNTRIES)]
        under_10k = ctxns[(ctxns['amount'] >= 7000) & (ctxns['amount'] < 10000)]
        crypto = ctxns[ctxns['method'] == 'crypto_exchange'] if 'method' in ctxns.columns else pd.DataFrame()
        num_currencies = ctxns['currency'].nunique() if 'currency' in ctxns.columns else 1

        sorted_txns = ctxns.sort_values('transaction_date')
        time_diffs = sorted_txns['transaction_date'].diff()
        rapid = (time_diffs < pd.Timedelta(days=2)).sum() if len(time_diffs) > 1 else 0
        avg_days = 0
        if len(time_diffs.dropna()) > 0:
            avg_td = time_diffs.dropna().mean()
            avg_days = avg_td.days if hasattr(avg_td, 'days') else 0

        # PPP asymmetry: log(PLI_receiver / PLI_sender) per transaction, then averaged.
        # Positive = cheap→expensive flow. Negative = expensive→cheap.
        # Only meaningful for international wire transactions.
        cust_ppl = COUNTRY_PPL.get(customer.get('country', 'USA'), DEFAULT_PPL)
        ppp_asymmetries = []
        for _, txn in intl[intl['method'] == 'wire'].iterrows():
            cp_country = txn.get('country', 'USA')
            cp_ppl = COUNTRY_PPL.get(cp_country, DEFAULT_PPL)
            if txn['transaction_type'] == 'deposit':
                # money coming in: sender=counterparty, receiver=customer
                sender_ppl, receiver_ppl = cp_ppl, cust_ppl
            else:
                # money going out: sender=customer, receiver=counterparty
                sender_ppl, receiver_ppl = cust_ppl, cp_ppl
            if sender_ppl > 0:
                ppp_asymmetries.append(np.log(receiver_ppl / sender_ppl))
        avg_ppp_asymmetry = float(np.mean(ppp_asymmetries)) if ppp_asymmetries else 0.0
        # USD-converted total volume using the daily usd_rate stored at
        # generation time (covers both FX and crypto correctly).
        if 'usd_rate' in ctxns.columns:
            usd_volume = (ctxns['amount'] * ctxns['usd_rate']).sum()
        else:
            usd_volume = ctxns['amount'].sum()

        features_list.append({
            'customer_id': customer['customer_id'],
            'customer_type_business': 1 if customer['customer_type'] == 'business' else 0,
            'annual_income': customer['annual_income'],
            'account_age_days': (pd.Timestamp('2025-12-31') - pd.to_datetime(customer['account_open_date'])).days,
            'total_transactions': len(ctxns),
            'total_volume': ctxns['amount'].sum(),
            'avg_transaction_amount': ctxns['amount'].mean(),
            'max_transaction_amount': ctxns['amount'].max(),
            'std_transaction_amount': ctxns['amount'].std() if len(ctxns) > 1 else 0,
            'transaction_date_range_days': date_range,
            'num_deposits': len(deposits),
            'num_withdrawals': len(withdrawals),
            'deposit_withdrawal_ratio': len(deposits) / (len(withdrawals) + 1),
            'total_deposits': deposits['amount'].sum() if len(deposits) > 0 else 0,
            'total_withdrawals': withdrawals['amount'].sum() if len(withdrawals) > 0 else 0,
            'num_cash_txns': len(cash),
            'num_wire_txns': len(wire),
            'pct_cash': len(cash) / len(ctxns),
            'pct_wire': len(wire) / len(ctxns),
            'total_cash_amount': cash['amount'].sum() if len(cash) > 0 else 0,
            'num_intl_txns': len(intl),
            'num_high_risk_country_txns': len(high_risk),
            'pct_intl': len(intl) / len(ctxns),
            'total_intl_amount': intl['amount'].sum() if len(intl) > 0 else 0,
            'num_just_under_10k': len(under_10k),
            'pct_just_under_10k': len(under_10k) / len(ctxns),
            'num_rapid_sequence_txns': rapid,
            'avg_days_between_txns': avg_days,
            'volume_to_income_ratio': usd_volume / (customer['annual_income'] + 1),
            'usd_volume': usd_volume,
            # PPP asymmetry: avg log(PLI_receiver / PLI_sender) across intl wires.
            # Positive = money flowing from cheap to expensive countries (suspicious).
            # Negative = expensive to cheap (e.g. remittance, less suspicious).
            # Zero = domestic only.
            'avg_ppp_asymmetry': round(avg_ppp_asymmetry, 4),
            'num_unique_counterparties': ctxns['counterparty'].nunique(),
            'num_unique_locations': ctxns['location'].nunique(),
            # Crypto-specific features
            'num_crypto_txns': len(crypto),
            'total_crypto_volume': crypto['amount'].sum() if len(crypto) > 0 else 0,
            'num_currencies_used': num_currencies,
            'crypto_volume_to_income_ratio': (crypto['amount'].sum() / (customer['annual_income'] + 1)) if len(crypto) > 0 else 0,
            'is_suspicious': int(customer['is_suspicious']),
            'typology': customer['typology'] if customer['is_suspicious'] else 'normal',
        })

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1:,} / {len(customers_df):,}")

    return pd.DataFrame(features_list)


def merge_all_features(traditional_df, behavioral_df, graph_df, bsi_df):
    """Merge traditional, behavioral, graph, and BSI features."""
    print("\nMerging feature sets...")
    merged = traditional_df.copy()

    # Merge behavioral signals
    beh_cols = [c for c in behavioral_df.columns if c != 'customer_id']
    merged = merged.merge(behavioral_df[['customer_id'] + beh_cols], on='customer_id', how='left')

    # Merge graph signals
    graph_cols = [c for c in graph_df.columns if c != 'customer_id']
    merged = merged.merge(graph_df[['customer_id'] + graph_cols], on='customer_id', how='left')

    # Merge BSI
    bsi_cols = ['bsi_score', 'entropy_stability', 'temporal_stability',
                'counterparty_stability', 'amount_stability', 'network_stability']
    bsi_merge = bsi_df[['customer_id'] + [c for c in bsi_cols if c in bsi_df.columns]]
    merged = merged.merge(bsi_merge, on='customer_id', how='left')

    # Fill NaN with 0
    merged = merged.fillna(0)

    print(f"  Total features: {len(merged.columns) - 3}")  # exclude id, suspicious, typology
    return merged


def train_meta_classifier(features_df):
    """Train XGBoost meta-classifier over all signal features."""
    print("\nTraining meta-risk classifier...")

    exclude_cols = ['customer_id', 'is_suspicious', 'typology', 'risk_score', 'drift_level']
    # Also exclude boolean columns that need conversion
    bool_cols = features_df.select_dtypes(include=['bool']).columns.tolist()

    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)

    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    X = features_df[feature_cols].values
    y = features_df['is_suspicious'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Suspicious']))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    # Risk scores for all
    features_df['risk_score'] = model.predict_proba(X)[:, 1]

    # SHAP
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'risk_classifier.pkl'))
    joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    print("[OK] Model and explainer saved")

    return features_df, model, explainer, feature_cols


def save_enriched_alerts(features_df, alerts_df, db_path=DB_PATH):
    """Update alerts with risk scores."""
    print("\nSaving enriched alerts...")
    conn = sqlite3.connect(db_path)

    score_df = features_df[features_df['is_suspicious'] == 1][['customer_id', 'risk_score']]
    enriched = alerts_df.merge(score_df, on='customer_id', how='left')
    enriched.to_sql('alerts', conn, if_exists='replace', index=False)

    conn.close()
    print("[OK] Alerts updated with risk scores")
    return enriched


def run(customers_df, transactions_df, alerts_df, behavioral_df, graph_df, bsi_df):
    """Full risk scoring pipeline."""
    traditional_df = engineer_traditional_features(customers_df, transactions_df)
    features_df = merge_all_features(traditional_df, behavioral_df, graph_df, bsi_df)
    features_df, model, explainer, feature_cols = train_meta_classifier(features_df)
    enriched_alerts = save_enriched_alerts(features_df, alerts_df)
    features_df.to_csv(os.path.join(os.path.dirname(MODELS_DIR), 'alert_features_scored.csv'), index=False)
    return features_df, model, explainer, feature_cols, enriched_alerts
