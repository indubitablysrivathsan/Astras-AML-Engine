"""
Adaptive Monitoring System
Dynamically adjusts monitoring intensity based on Behavioral Stability Index.
Transforms AML from static rule enforcement to dynamic behavioral surveillance.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MONITORING_LEVELS, BSI_CRITICAL, BSI_HIGH_DRIFT, BSI_MODERATE


def determine_monitoring_level(bsi_score, previous_bsi=None):
    """
    Determine monitoring intensity based on BSI.
    Also detects sudden stability drops for automatic escalation.
    """
    # Sudden drop detection
    sudden_drop = False
    if previous_bsi is not None:
        drop = previous_bsi - bsi_score
        if drop > 20:  # >20 point drop = sudden stability loss
            sudden_drop = True

    if bsi_score <= BSI_CRITICAL or sudden_drop:
        level = 'immediate'
        action = 'Immediate investigation warranted. SAR review initiated.'
    elif bsi_score <= BSI_HIGH_DRIFT:
        level = 'intensive'
        action = 'Enhanced monitoring required. Lower anomaly thresholds.'
    elif bsi_score <= BSI_MODERATE:
        level = 'enhanced'
        action = 'Review transaction context and patterns.'
    else:
        level = 'standard'
        action = 'Behavioral consistency maintained. Low risk.'

    config = MONITORING_LEVELS[level]
    return {
        'monitoring_level': level,
        'check_interval_days': config['check_interval_days'],
        'anomaly_threshold': config['anomaly_threshold'],
        'recommended_action': action,
        'sudden_drop_detected': sudden_drop,
        'escalation_triggered': level in ('immediate', 'intensive') or sudden_drop,
    }


def compute_monitoring_states(bsi_df):
    """Compute monitoring state for all customers."""
    print("\nComputing adaptive monitoring states...")
    results = []

    for _, row in bsi_df.iterrows():
        state = determine_monitoring_level(row['bsi_score'])
        state['customer_id'] = row['customer_id']
        state['bsi_score'] = row['bsi_score']
        state['drift_level'] = row['drift_level']
        state['timestamp'] = datetime.now().isoformat()
        results.append(state)

    monitor_df = pd.DataFrame(results)

    escalated = monitor_df[monitor_df['escalation_triggered']].shape[0]
    print(f"  Escalations triggered: {escalated} / {len(monitor_df)}")
    print(f"  Monitoring levels: {monitor_df['monitoring_level'].value_counts().to_dict()}")

    return monitor_df
