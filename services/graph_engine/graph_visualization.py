"""
Graph Visualization Service
Converts NetworkX graphs to Plotly interactive visualizations.
"""
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from services.graph_engine import build_transaction_graph


def create_customer_graph_figure(customer_id, transactions_df, max_nodes=30):
    """
    Create an interactive Plotly figure of the customer's transaction network.
    Shows the customer node, counterparties, and edges weighted by amount.
    """
    ctxns = transactions_df[transactions_df['customer_id'] == customer_id].copy()
    if len(ctxns) == 0:
        return None

    # Build a local graph for this customer
    G = nx.DiGraph()
    customer_node = f"Customer\n{customer_id}"
    G.add_node(customer_node, node_type='customer')

    # Aggregate by counterparty
    counterparty_stats = {}
    for _, txn in ctxns.iterrows():
        cp = txn.get('counterparty')
        if pd.isna(cp) or cp == '' or cp is None:
            continue
        cp = str(cp)[:20]  # truncate long names
        if cp not in counterparty_stats:
            counterparty_stats[cp] = {
                'inflow': 0, 'outflow': 0, 'count': 0,
                'country': txn.get('country', 'USA'),
                'method': txn.get('method', 'unknown'),
            }
        if txn['transaction_type'] == 'deposit':
            counterparty_stats[cp]['inflow'] += txn['amount']
        else:
            counterparty_stats[cp]['outflow'] += txn['amount']
        counterparty_stats[cp]['count'] += 1

    # Take top counterparties by total volume
    sorted_cps = sorted(counterparty_stats.items(),
                        key=lambda x: x[1]['inflow'] + x[1]['outflow'],
                        reverse=True)[:max_nodes]

    for cp_name, stats in sorted_cps:
        G.add_node(cp_name, node_type='counterparty', country=stats['country'])
        if stats['inflow'] > 0:
            G.add_edge(cp_name, customer_node, weight=stats['inflow'],
                       flow_type='inflow', count=stats['count'])
        if stats['outflow'] > 0:
            G.add_edge(customer_node, cp_name, weight=stats['outflow'],
                       flow_type='outflow', count=stats['count'])

    if G.number_of_nodes() < 2:
        return None

    # Layout
    pos = nx.spring_layout(G, k=2, seed=42)

    # Edge traces
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 0)
        flow_type = data.get('flow_type', 'unknown')

        # Line width proportional to log of amount
        width = max(1, min(8, np.log10(weight + 1)))
        color = '#ef4444' if flow_type == 'outflow' else '#22c55e'

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"{'→' if flow_type == 'outflow' else '←'} ${weight:,.0f} ({data.get('count', 0)} txns)",
            showlegend=False,
        ))

    # Node traces
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        ntype = G.nodes[node].get('node_type', 'counterparty')
        if ntype == 'customer':
            node_color.append('#3b82f6')
            node_size.append(25)
            node_text.append(f"<b>{node}</b><br>(Primary Account)")
        else:
            country = G.nodes[node].get('country', 'USA')
            is_intl = country != 'USA'
            node_color.append('#f97316' if is_intl else '#64748b')
            node_size.append(15)
            stats = counterparty_stats.get(node.replace('\n', ''), {})
            if not stats:
                # Try without truncation
                for cp_name, s in counterparty_stats.items():
                    if cp_name[:20] == node:
                        stats = s
                        break
            node_text.append(
                f"<b>{node}</b><br>"
                f"Country: {country}<br>"
                f"In: ${stats.get('inflow', 0):,.0f}<br>"
                f"Out: ${stats.get('outflow', 0):,.0f}"
            )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        text=[n.split('\n')[0][:15] for n in G.nodes()],
        textposition='top center',
        textfont=dict(size=8),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False,
    )

    # Build figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=500,
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(text="🟢 Inflow  🔴 Outflow  🔵 Customer  🟠 International",
                 xref="paper", yref="paper", x=0.5, y=-0.02,
                 showarrow=False, font=dict(size=10))
        ]
    )
    return fig
