"""
Behavioral Stability Index (BSI)
Aggregates signals from 5 behavioral dimensions into a single stability score (0-100).
Money laundering is often not about abnormal behavior - it's about sudden behavioral transition.

Scoring: 100 = perfectly stable, 0 = extreme behavioral drift.
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BSI_CRITICAL, BSI_HIGH_DRIFT, BSI_MODERATE, BSI_STABLE


def normalize_signal(value, min_val=0, max_val=1):
    """Normalize a signal to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return np.clip((value - min_val) / (max_val - min_val), 0, 1)


def compute_bsi(behavioral_signals, graph_signals):
    """
    Compute BSI from 5 signal dimensions.
    Uses population-calibrated thresholds so suspicious customers
    actually land in critical/high drift zones.
    """
    # Dimension 1: Entropy stability (lower drift = more stable)
    # Calibrated: normal customers drift ~0.1-0.4, suspicious ~0.5-2.0+
    entropy_drift = behavioral_signals.get('amount_entropy_drift', 0)
    timing_drift = behavioral_signals.get('timing_entropy_drift', 0)
    cp_drift = behavioral_signals.get('counterparty_entropy_drift', 0)
    avg_drift = (entropy_drift + timing_drift + cp_drift) / 3
    entropy_stability = 1 - normalize_signal(avg_drift, 0, 2.0)

    # Dimension 2: Temporal regularity
    burstiness = behavioral_signals.get('burstiness_index', 0)
    burst_events = behavioral_signals.get('burst_events', 0)
    max_daily = behavioral_signals.get('max_daily_count', 0)

    # Positive burstiness = bursty, negative = regular
    temporal_score = 1 - normalize_signal(max(burstiness, 0), 0, 1.0)
    burst_penalty = normalize_signal(burst_events, 0, 30)
    daily_penalty = normalize_signal(max_daily, 0, 15)
    temporal_stability = temporal_score * 0.4 + (1 - burst_penalty) * 0.3 + (1 - daily_penalty) * 0.3

    # Dimension 3: Counterparty stability
    expansion = behavioral_signals.get('counterparty_expansion_rate', 0)
    novelty = behavioral_signals.get('counterparty_novelty_slope', 0)
    total_cps = behavioral_signals.get('total_unique_counterparties', 0)

    cp_stability = 1 - normalize_signal(expansion, 0, 15)
    novelty_factor = 1 - normalize_signal(abs(novelty), 0, 3)
    cp_count_factor = 1 - normalize_signal(total_cps, 0, 80)
    counterparty_stability = cp_stability * 0.4 + novelty_factor * 0.3 + cp_count_factor * 0.3

    # Dimension 4: Amount normality
    benford = behavioral_signals.get('benford_deviation', 0)
    structuring = behavioral_signals.get('structuring_score', 0)
    skewness = abs(behavioral_signals.get('amount_skewness', 0))
    kurtosis = abs(behavioral_signals.get('amount_kurtosis', 0))

    amount_stability = 1 - (
        normalize_signal(benford, 0, 1.5) * 0.3 +
        normalize_signal(structuring, 0, 0.3) * 0.3 +
        normalize_signal(skewness, 0, 5) * 0.2 +
        normalize_signal(kurtosis, 0, 20) * 0.2
    )

    # Dimension 5: Network topology stability
    funnel_ratio = graph_signals.get('funnel_ratio', 1)
    layer_depth = graph_signals.get('layer_depth', 0)
    flow_velocity = graph_signals.get('flow_velocity', 0)
    has_circular = 1 if graph_signals.get('has_circular_flow', False) else 0
    rapid_passthrough = graph_signals.get('rapid_passthrough_count', 0)
    is_funnel = 1 if graph_signals.get('is_funnel_hub', False) else 0

    network_stability = 1 - (
        normalize_signal(funnel_ratio, 1, 8) * 0.2 +
        normalize_signal(layer_depth, 0, 6) * 0.2 +
        normalize_signal(flow_velocity, 0, 0.3) * 0.15 +
        has_circular * 0.15 +
        is_funnel * 0.15 +
        normalize_signal(rapid_passthrough, 0, 10) * 0.15
    )

    # Composite BSI: weighted average
    weights = {
        'entropy': 0.25,
        'temporal': 0.20,
        'counterparty': 0.20,
        'amount': 0.20,
        'network': 0.15,
    }

    bsi_raw = (
        entropy_stability * weights['entropy'] +
        temporal_stability * weights['temporal'] +
        counterparty_stability * weights['counterparty'] +
        amount_stability * weights['amount'] +
        network_stability * weights['network']
    )

    # Scale to 0-100
    bsi_score = round(bsi_raw * 100, 2)

    if bsi_score <= BSI_CRITICAL:
        drift_level = 'critical'
    elif bsi_score <= BSI_HIGH_DRIFT:
        drift_level = 'high'
    elif bsi_score <= BSI_MODERATE:
        drift_level = 'moderate'
    else:
        drift_level = 'stable'

    return {
        'bsi_score': bsi_score,
        'drift_level': drift_level,
        'entropy_stability': round(entropy_stability * 100, 2),
        'temporal_stability': round(temporal_stability * 100, 2),
        'counterparty_stability': round(counterparty_stability * 100, 2),
        'amount_stability': round(amount_stability * 100, 2),
        'network_stability': round(network_stability * 100, 2),
    }


def compute_bsi_for_all(behavioral_df, graph_df):
    """Compute BSI for all customers with population-calibrated rescaling."""
    print("\nComputing BSI scores...")
    merged = behavioral_df.merge(graph_df, on='customer_id', how='inner')

    results = []
    for _, row in merged.iterrows():
        signals = row.to_dict()
        bsi = compute_bsi(signals, signals)
        bsi['customer_id'] = row['customer_id']
        results.append(bsi)

    bsi_df = pd.DataFrame(results)

    # Population-calibrated rescaling: spread raw scores across 0-100
    # This ensures the distribution reflects relative behavioral differences
    raw = bsi_df['bsi_score']
    p5, p95 = raw.quantile(0.05), raw.quantile(0.95)
    if p95 > p5:
        rescaled = ((raw - p5) / (p95 - p5) * 90 + 5).clip(2, 98)
        bsi_df['bsi_score'] = rescaled.round(2)

        # Recompute drift levels with rescaled scores
        bsi_df['drift_level'] = bsi_df['bsi_score'].apply(
            lambda s: 'critical' if s <= BSI_CRITICAL
            else 'high' if s <= BSI_HIGH_DRIFT
            else 'moderate' if s <= BSI_MODERATE
            else 'stable'
        )

    print(f"  BSI computed for {len(bsi_df)} customers")
    print(f"  Distribution: {bsi_df['drift_level'].value_counts().to_dict()}")
    return bsi_df
