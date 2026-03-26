"""
BSI Timeline Service
Computes BSI scores across rolling time windows to show
behavioral regime changes over time.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.behavioral_engine.behavioral_signals import compute_all_signals
from services.graph_engine.graph_core import build_transaction_graph, compute_graph_signals
from services.behavioral_engine.bsi import compute_bsi


def compute_bsi_timeline(customer_id, transactions_df, window_days=60, step_days=30):
    """
    Compute BSI at multiple time points to create a timeline.
    Uses sliding windows of `window_days` with `step_days` between each.
    """
    ctxns = transactions_df[transactions_df['customer_id'] == customer_id].copy()
    if len(ctxns) < 5:
        return pd.DataFrame()

    ctxns['transaction_date'] = pd.to_datetime(ctxns['transaction_date'])
    min_date = ctxns['transaction_date'].min()
    max_date = ctxns['transaction_date'].max()

    timeline = []
    current = min_date

    while current + timedelta(days=window_days) <= max_date + timedelta(days=1):
        end = current + timedelta(days=window_days)
        window = ctxns[(ctxns['transaction_date'] >= current) & (ctxns['transaction_date'] < end)]

        if len(window) >= 3:
            # Compute signals for this window
            beh = compute_all_signals(customer_id, window)
            G = build_transaction_graph(window)
            graph = compute_graph_signals(customer_id, window, G)
            bsi = compute_bsi(beh, graph)

            timeline.append({
                'window_start': current,
                'window_end': end,
                'window_midpoint': current + timedelta(days=window_days // 2),
                'num_transactions': len(window),
                'total_volume': window['amount'].sum(),
                **bsi,
            })

        current += timedelta(days=step_days)

    return pd.DataFrame(timeline)
