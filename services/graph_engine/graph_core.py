"""
NetworkX Graph Processing Engine
Detects network structural anomalies: circular flows, funnel/mule hubs,
layering depth, and flow velocity.
"""
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_transaction_graph(transactions_df):
    """Build a directed weighted graph from transaction data."""
    G = nx.DiGraph()

    for _, txn in transactions_df.iterrows():
        sender = str(txn['customer_id'])
        receiver = txn.get('counterparty')
        if pd.isna(receiver) or receiver == '' or receiver is None:
            continue

        receiver = str(receiver)
        amount = txn['amount']
        date = txn['transaction_date']

        if G.has_edge(sender, receiver):
            G[sender][receiver]['weight'] += amount
            G[sender][receiver]['count'] += 1
            G[sender][receiver]['dates'].append(date)
        else:
            G.add_edge(sender, receiver, weight=amount, count=1, dates=[date])

        # Add reverse edge for withdrawals going to counterparty
        if txn['transaction_type'] == 'withdrawal':
            if G.has_edge(receiver, sender):
                pass  # Already tracked
            # Mark this node as receiver of funds
            if not G.has_node(receiver):
                G.add_node(receiver, node_type='counterparty')
            if not G.has_node(sender):
                G.add_node(sender, node_type='customer')

    return G


def detect_circular_flows(G, customer_id):
    """Detect circular fund flows involving the customer."""
    node = str(customer_id)
    if node not in G:
        return {'has_circular_flow': False, 'num_cycles': 0, 'max_cycle_length': 0,
                'cycle_total_amount': 0.0}

    cycles = []
    try:
        # Find simple cycles involving this node (limit cycle length)
        for cycle in nx.simple_cycles(G):
            if node in cycle and len(cycle) <= 8:
                cycles.append(cycle)
            if len(cycles) >= 20:
                break
    except Exception:
        pass

    if not cycles:
        return {'has_circular_flow': False, 'num_cycles': 0, 'max_cycle_length': 0,
                'cycle_total_amount': 0.0}

    max_len = max(len(c) for c in cycles)
    total_amount = 0
    for cycle in cycles:
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if G.has_edge(u, v):
                total_amount += G[u][v].get('weight', 0)

    return {
        'has_circular_flow': True,
        'num_cycles': len(cycles),
        'max_cycle_length': max_len,
        'cycle_total_amount': float(total_amount),
    }


def detect_funnel_hub(G, customer_id):
    """Detect funnel/mule hub patterns: many inputs, few outputs (or vice versa)."""
    node = str(customer_id)
    if node not in G:
        return {'is_funnel_hub': False, 'in_degree': 0, 'out_degree': 0,
                'funnel_ratio': 0.0, 'total_inflow': 0.0, 'total_outflow': 0.0}

    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)

    total_inflow = sum(G[u][node].get('weight', 0) for u in G.predecessors(node))
    total_outflow = sum(G[node][v].get('weight', 0) for v in G.successors(node))

    # Funnel ratio: high in_degree with low out_degree or vice versa
    max_degree = max(in_degree, out_degree)
    min_degree = min(in_degree, out_degree) + 1  # avoid /0
    funnel_ratio = max_degree / min_degree

    is_funnel = (in_degree >= 4 and out_degree <= 2) or (out_degree >= 4 and in_degree <= 2)

    return {
        'is_funnel_hub': is_funnel,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'funnel_ratio': float(funnel_ratio),
        'total_inflow': float(total_inflow),
        'total_outflow': float(total_outflow),
    }


def compute_layer_depth(G, customer_id):
    """Compute the depth of layering from this customer node."""
    node = str(customer_id)
    if node not in G:
        return {'layer_depth': 0, 'reachable_nodes': 0}

    try:
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=10)
        max_depth = max(lengths.values()) if lengths else 0
        reachable = len(lengths) - 1  # exclude self
    except Exception:
        max_depth = 0
        reachable = 0

    return {
        'layer_depth': max_depth,
        'reachable_nodes': reachable,
    }


def compute_flow_velocity(transactions_df, customer_id):
    """
    Compute flow velocity: how quickly funds move through the account.
    Measured as the ratio of outflow to inflow within short time windows.
    """
    ctxns = transactions_df[transactions_df['customer_id'] == customer_id].copy()
    ctxns['transaction_date'] = pd.to_datetime(ctxns['transaction_date'])
    ctxns = ctxns.sort_values('transaction_date')

    if len(ctxns) < 2:
        return {'flow_velocity': 0.0, 'avg_retention_hours': float('inf'),
                'rapid_passthrough_count': 0}

    deposits = ctxns[ctxns['transaction_type'] == 'deposit']
    withdrawals = ctxns[ctxns['transaction_type'] == 'withdrawal']

    if len(deposits) == 0 or len(withdrawals) == 0:
        return {'flow_velocity': 0.0, 'avg_retention_hours': float('inf'),
                'rapid_passthrough_count': 0}

    # For each deposit, find the nearest following withdrawal
    retention_times = []
    rapid_count = 0

    for _, dep in deposits.iterrows():
        subsequent = withdrawals[withdrawals['transaction_date'] > dep['transaction_date']]
        if len(subsequent) > 0:
            first_wd = subsequent.iloc[0]
            retention_hrs = (first_wd['transaction_date'] - dep['transaction_date']).total_seconds() / 3600
            retention_times.append(retention_hrs)
            if retention_hrs < 72:  # Less than 3 days
                rapid_count += 1

    if retention_times:
        avg_retention = np.mean(retention_times)
        # Flow velocity: inverse of retention time (higher = faster passthrough)
        flow_velocity = 1.0 / (avg_retention + 1)
    else:
        avg_retention = float('inf')
        flow_velocity = 0.0

    return {
        'flow_velocity': float(flow_velocity),
        'avg_retention_hours': float(avg_retention) if avg_retention != float('inf') else 999999.0,
        'rapid_passthrough_count': rapid_count,
    }


def compute_graph_signals(customer_id, transactions_df, G=None,pagerank=None):
    """Compute all graph-based signals for a customer."""
    if G is None:
        ctxns = transactions_df[transactions_df['customer_id'] == customer_id]
        G = build_transaction_graph(ctxns)

    circular = detect_circular_flows(G, customer_id)
    funnel = detect_funnel_hub(G, customer_id)
    depth = compute_layer_depth(G, customer_id)
    velocity = compute_flow_velocity(transactions_df, customer_id)
    pr = compute_pagerank_score(G, customer_id, pagerank)

    return {
        'customer_id': customer_id,
        # Circular flow signals
        'has_circular_flow': circular['has_circular_flow'],
        'num_cycles': circular['num_cycles'],
        'max_cycle_length': circular['max_cycle_length'],
        'cycle_total_amount': circular['cycle_total_amount'],
        # Funnel hub signals
        'is_funnel_hub': funnel['is_funnel_hub'],
        'in_degree': funnel['in_degree'],
        'out_degree': funnel['out_degree'],
        'funnel_ratio': funnel['funnel_ratio'],
        'total_inflow': funnel['total_inflow'],
        'total_outflow': funnel['total_outflow'],
        # Layer depth
        'layer_depth': depth['layer_depth'],
        'reachable_nodes': depth['reachable_nodes'],
        # Flow velocity
        'flow_velocity': velocity['flow_velocity'],
        'avg_retention_hours': velocity['avg_retention_hours'],
        'rapid_passthrough_count': velocity['rapid_passthrough_count'],
        # Pagerank
        'pagerank_score': pr['pagerank_score'],
    }

def compute_pagerank_score(G, customer_id, pagerank_dict):
    node = str(customer_id)
    return {
        'pagerank_score': float(pagerank_dict.get(node, 0.0))
    }


def compute_graph_signals_for_all(customers_df, transactions_df, G=None, pagerank=None):
    """Compute graph signals for all customers."""
    print("\nComputing graph signals...")

    # Build global graph
    G = build_transaction_graph(transactions_df)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pagerank = nx.pagerank(G, weight='weight')

    all_signals = []
    for idx, customer in customers_df.iterrows():
        signals = compute_graph_signals(customer['customer_id'], transactions_df, G, pagerank)
        all_signals.append(signals)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1:,} / {len(customers_df):,} customers")

    return pd.DataFrame(all_signals)