"""
Behavioral Signal Engines
Computes atomic anomaly signals: entropy drift, temporal burst patterns,
counterparty expansion, and amount distribution irregularities.
"""
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROLLING_WINDOW_DAYS


def compute_entropy(values):
    """Shannon entropy of a discrete distribution."""
    if len(values) == 0:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def compute_amount_entropy(amounts, num_bins=10):
    """Entropy of the amount distribution (binned)."""
    if len(amounts) < 2:
        return 0.0
    bins = np.linspace(amounts.min(), amounts.max() + 1, num_bins + 1)
    digitized = np.digitize(amounts, bins)
    return compute_entropy(digitized)


def compute_timing_entropy(timestamps, num_bins=7):
    """Entropy of transaction timing (day-of-week distribution)."""
    if len(timestamps) < 2:
        return 0.0
    days = timestamps.dt.dayofweek
    return compute_entropy(days.values)


def compute_counterparty_entropy(counterparties):
    """Entropy of counterparty distribution."""
    valid = [c for c in counterparties if pd.notna(c) and c != '']
    if len(valid) < 2:
        return 0.0
    return compute_entropy(valid)


def compute_entropy_drift(txns, window_days=ROLLING_WINDOW_DAYS):
    """
    Compute entropy drift: change in entropy between consecutive windows.
    Returns per-window entropy values and the drift (delta).
    """
    txns = txns.sort_values('transaction_date').copy()
    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'])

    min_date = txns['transaction_date'].min()
    max_date = txns['transaction_date'].max()
    total_days = (max_date - min_date).days
    if total_days < window_days * 2:
        amt_e = compute_amount_entropy(txns['amount'].values)
        tim_e = compute_timing_entropy(txns['transaction_date'])
        cp_e = compute_counterparty_entropy(
            txns['counterparty'].values if 'counterparty' in txns.columns else [])
        return {
            'amount_entropy': [amt_e], 'timing_entropy': [tim_e], 'counterparty_entropy': [cp_e],
            'amount_entropy_drift': 0.0, 'timing_entropy_drift': 0.0,
            'counterparty_entropy_drift': 0.0,
            'mean_amount_entropy': amt_e, 'mean_timing_entropy': tim_e,
            'mean_counterparty_entropy': cp_e,
            'entropy_drift': 0.0, 'num_windows': 1,
        }

    windows = []
    current = min_date
    while current + pd.Timedelta(days=window_days) <= max_date:
        end = current + pd.Timedelta(days=window_days)
        window_txns = txns[(txns['transaction_date'] >= current) & (txns['transaction_date'] < end)]
        if len(window_txns) > 0:
            windows.append({
                'start': current,
                'end': end,
                'amount_entropy': compute_amount_entropy(window_txns['amount'].values),
                'timing_entropy': compute_timing_entropy(window_txns['transaction_date']),
                'counterparty_entropy': compute_counterparty_entropy(
                    window_txns['counterparty'].values if 'counterparty' in window_txns.columns else []),
            })
        current = end

    if len(windows) < 2:
        ae = windows[0]['amount_entropy'] if windows else 0.0
        te = windows[0]['timing_entropy'] if windows else 0.0
        ce = windows[0]['counterparty_entropy'] if windows else 0.0
        return {
            'amount_entropy': [ae], 'timing_entropy': [te], 'counterparty_entropy': [ce],
            'amount_entropy_drift': 0.0, 'timing_entropy_drift': 0.0,
            'counterparty_entropy_drift': 0.0,
            'mean_amount_entropy': ae, 'mean_timing_entropy': te,
            'mean_counterparty_entropy': ce,
            'entropy_drift': 0.0, 'num_windows': len(windows),
        }

    amt_entropies = [w['amount_entropy'] for w in windows]
    timing_entropies = [w['timing_entropy'] for w in windows]
    cp_entropies = [w['counterparty_entropy'] for w in windows]

    # Drift = max absolute change between consecutive windows
    amt_drifts = [abs(amt_entropies[i] - amt_entropies[i - 1]) for i in range(1, len(amt_entropies))]
    timing_drifts = [abs(timing_entropies[i] - timing_entropies[i - 1]) for i in range(1, len(timing_entropies))]
    cp_drifts = [abs(cp_entropies[i] - cp_entropies[i - 1]) for i in range(1, len(cp_entropies))]

    return {
        'amount_entropy': amt_entropies,
        'timing_entropy': timing_entropies,
        'counterparty_entropy': cp_entropies,
        'amount_entropy_drift': max(amt_drifts) if amt_drifts else 0.0,
        'timing_entropy_drift': max(timing_drifts) if timing_drifts else 0.0,
        'counterparty_entropy_drift': max(cp_drifts) if cp_drifts else 0.0,
        'mean_amount_entropy': np.mean(amt_entropies),
        'mean_timing_entropy': np.mean(timing_entropies),
        'mean_counterparty_entropy': np.mean(cp_entropies),
        'entropy_drift': max(amt_drifts) if amt_drifts else 0.0,
        'num_windows': len(windows),
    }


def compute_temporal_burst(txns, window_days=ROLLING_WINDOW_DAYS):
    """
    Detect temporal burst patterns: sudden spikes in transaction frequency.
    Returns burstiness index and burst event count.
    """
    txns = txns.sort_values('transaction_date').copy()
    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'])

    if len(txns) < 3:
        return {'burstiness_index': 0.0, 'burst_events': 0, 'inter_arrival_variance': 0.0,
                'max_daily_count': 0, 'mean_daily_count': 0.0}

    # Inter-arrival times in hours
    diffs = txns['transaction_date'].diff().dropna().dt.total_seconds() / 3600
    if len(diffs) == 0:
        return {'burstiness_index': 0.0, 'burst_events': 0, 'inter_arrival_variance': 0.0,
                'max_daily_count': 0, 'mean_daily_count': 0.0}

    mean_iat = diffs.mean()
    std_iat = diffs.std()

    # Burstiness index: (std - mean) / (std + mean), ranges from -1 to 1
    if (std_iat + mean_iat) > 0:
        burstiness = (std_iat - mean_iat) / (std_iat + mean_iat)
    else:
        burstiness = 0.0

    # Count burst events (transactions within 24 hours of each other in clusters)
    burst_threshold = 24  # hours
    burst_events = (diffs < burst_threshold).sum()

    # Daily transaction counts
    daily = txns.groupby(txns['transaction_date'].dt.date).size()
    max_daily = daily.max()
    mean_daily = daily.mean()

    return {
        'burstiness_index': float(burstiness),
        'burst_events': int(burst_events),
        'inter_arrival_variance': float(diffs.var()) if len(diffs) > 1 else 0.0,
        'max_daily_count': int(max_daily),
        'mean_daily_count': float(mean_daily),
    }


def compute_counterparty_expansion(txns, window_days=ROLLING_WINDOW_DAYS):
    """
    Measure counterparty network expansion rate.
    Tracks how quickly new counterparties appear over time.
    """
    txns = txns.sort_values('transaction_date').copy()
    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'])

    if 'counterparty' not in txns.columns or len(txns) < 2:
        return {'expansion_rate': 0.0, 'novelty_slope': 0.0, 'total_unique_counterparties': 0,
                'new_counterparties_per_window': []}

    valid_txns = txns[txns['counterparty'].notna() & (txns['counterparty'] != '')]
    if len(valid_txns) < 2:
        return {'expansion_rate': 0.0, 'novelty_slope': 0.0, 'total_unique_counterparties': 0,
                'new_counterparties_per_window': []}

    min_date = valid_txns['transaction_date'].min()
    max_date = valid_txns['transaction_date'].max()

    seen = set()
    new_per_window = []
    current = min_date
    while current + pd.Timedelta(days=window_days) <= max_date + pd.Timedelta(days=1):
        end = current + pd.Timedelta(days=window_days)
        window = valid_txns[(valid_txns['transaction_date'] >= current) &
                            (valid_txns['transaction_date'] < end)]
        new_cps = set(window['counterparty'].unique()) - seen
        new_per_window.append(len(new_cps))
        seen.update(new_cps)
        current = end

    total_unique = len(seen)
    # Novelty slope: linear regression of new counterparties over windows
    if len(new_per_window) >= 2:
        x = np.arange(len(new_per_window))
        slope, _, _, _, _ = stats.linregress(x, new_per_window)
    else:
        slope = 0.0

    # Expansion rate: new counterparties per window on average
    expansion_rate = np.mean(new_per_window) if new_per_window else 0.0

    return {
        'expansion_rate': float(expansion_rate),
        'novelty_slope': float(slope),
        'total_unique_counterparties': total_unique,
        'new_counterparties_per_window': new_per_window,
    }


def compute_amount_anomaly(txns):
    """
    Detect amount distribution anomalies:
    - Benford's law deviation
    - Structuring detection (amounts just below thresholds)
    - Round amount ratio
    """
    if len(txns) < 5:
        return {'benford_deviation': 0.0, 'structuring_score': 0.0, 'round_amount_ratio': 0.0,
                'amount_skewness': 0.0, 'amount_kurtosis': 0.0}

    amounts = txns['amount'].values
    amounts = amounts[amounts > 0]

    if len(amounts) < 5:
        return {'benford_deviation': 0.0, 'structuring_score': 0.0, 'round_amount_ratio': 0.0,
                'amount_skewness': 0.0, 'amount_kurtosis': 0.0}

    # Benford's law: expected first-digit distribution
    first_digits = [int(str(abs(a)).replace('.', '').lstrip('0')[0]) for a in amounts if a > 0]
    first_digits = [d for d in first_digits if 1 <= d <= 9]

    benford_expected = {d: np.log10(1 + 1 / d) for d in range(1, 10)}
    if first_digits:
        observed = Counter(first_digits)
        total = sum(observed.values())
        deviation = sum(
            abs(observed.get(d, 0) / total - benford_expected[d])
            for d in range(1, 10)
        )
    else:
        deviation = 0.0

    # Structuring: fraction of amounts in [7000, 10000) range
    structuring_range = ((amounts >= 7000) & (amounts < 10000)).sum()
    structuring_score = structuring_range / len(amounts)

    # Round amounts (ending in 00 or 000)
    round_count = sum(1 for a in amounts if a % 100 == 0 or a % 1000 == 0)
    round_ratio = round_count / len(amounts)

    return {
        'benford_deviation': float(deviation),
        'structuring_score': float(structuring_score),
        'round_amount_ratio': float(round_ratio),
        'amount_skewness': float(stats.skew(amounts)) if len(amounts) > 2 else 0.0,
        'amount_kurtosis': float(stats.kurtosis(amounts)) if len(amounts) > 3 else 0.0,
    }


def compute_all_signals(customer_id, txns):
    """Compute all behavioral signals for a single customer."""
    txns = txns.copy()
    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'])

    entropy = compute_entropy_drift(txns)
    burst = compute_temporal_burst(txns)
    expansion = compute_counterparty_expansion(txns)
    amount = compute_amount_anomaly(txns)

    # Flatten into a single feature dict
    signals = {
        'customer_id': customer_id,
        # Entropy signals
        'amount_entropy_drift': entropy['amount_entropy_drift'],
        'timing_entropy_drift': entropy['timing_entropy_drift'],
        'counterparty_entropy_drift': entropy['counterparty_entropy_drift'],
        'mean_amount_entropy': entropy['mean_amount_entropy'],
        'mean_timing_entropy': entropy['mean_timing_entropy'],
        'mean_counterparty_entropy': entropy['mean_counterparty_entropy'],
        # Burst signals
        'burstiness_index': burst['burstiness_index'],
        'burst_events': burst['burst_events'],
        'inter_arrival_variance': burst['inter_arrival_variance'],
        'max_daily_count': burst['max_daily_count'],
        'mean_daily_count': burst['mean_daily_count'],
        # Counterparty expansion signals
        'counterparty_expansion_rate': expansion['expansion_rate'],
        'counterparty_novelty_slope': expansion['novelty_slope'],
        'total_unique_counterparties': expansion['total_unique_counterparties'],
        # Amount anomaly signals
        'benford_deviation': amount['benford_deviation'],
        'structuring_score': amount['structuring_score'],
        'round_amount_ratio': amount['round_amount_ratio'],
        'amount_skewness': amount['amount_skewness'],
        'amount_kurtosis': amount['amount_kurtosis'],
    }

    return signals


def compute_signals_for_all_customers(customers_df, transactions_df):
    """Compute behavioral signals for every customer."""
    print("\nComputing behavioral signals...")
    all_signals = []

    for idx, customer in customers_df.iterrows():
        ctxns = transactions_df[transactions_df['customer_id'] == customer['customer_id']]
        if len(ctxns) == 0:
            continue
        signals = compute_all_signals(customer['customer_id'], ctxns)
        all_signals.append(signals)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1:,} / {len(customers_df):,} customers")

    return pd.DataFrame(all_signals)
