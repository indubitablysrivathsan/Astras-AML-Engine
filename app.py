"""
ASTRAS - Enhanced Streamlit Dashboard
Behavioral Intelligence-Driven AML Detection & SAR Generation
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import json
import joblib
import glob
import os
import numpy as np
from datetime import datetime

from config import DB_PATH, MODELS_DIR, OUTPUTS_DIR, CHROMA_DIR, LLM_MODEL, HIGH_RISK_THRESHOLD

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="ASTRAS - SAR Generator", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    [data-testid="stMetric"] label { color: #94a3b8 !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    .bsi-critical { color: #ef4444; font-weight: bold; }
    .bsi-high { color: #f97316; font-weight: bold; }
    .bsi-moderate { color: #eab308; font-weight: bold; }
    .bsi-stable { color: #22c55e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────
@st.cache_resource
def load_all_data():
    conn = sqlite3.connect(DB_PATH)
    customers = pd.read_sql('SELECT * FROM customers', conn)
    transactions = pd.read_sql('SELECT * FROM transactions', conn)
    alerts = pd.read_sql('SELECT * FROM alerts', conn)
    conn.close()

    features_df = pd.read_csv('alert_features_scored.csv')
    explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

    # Load BSI and monitoring
    base = os.path.dirname(MODELS_DIR)
    bsi_path = os.path.join(base, 'bsi_scores.csv')
    bsi_df = pd.read_csv(bsi_path) if os.path.exists(bsi_path) else None

    monitor_path = os.path.join(base, 'monitoring_states.csv')
    monitor_df = pd.read_csv(monitor_path) if os.path.exists(monitor_path) else None

    # Warm up LLM — load model into VRAM once at startup, keep alive permanently.
    # This eliminates the cold-start delay on the first Investigation query.
    try:
        import requests
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": LLM_MODEL, "keep_alive": -1},
            timeout=5,
        )
    except Exception:
        pass  # Ollama not running — fine, Investigation page handles this gracefully

    return customers, transactions, alerts, features_df, explainer, feature_cols, bsi_df, monitor_df


try:
    customers_df, transactions_df, alerts_df, features_df, explainer, feature_cols, bsi_df, monitor_df = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Data not found. Run `python run_pipeline.py` first.\n\nError: {e}")
    data_loaded = False
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("ASTRAS")
st.sidebar.caption("Behavioral Intelligence-Driven\nSAR Generation System")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Alert Dashboard",
    "Behavioral Intelligence",
    "Generate SAR",
    "Investigation",
    "Counterfactual Analysis",
    "Audit Trail Viewer",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**System Info**")
st.sidebar.markdown(f"- Model: {LLM_MODEL}")
st.sidebar.markdown(f"- Customers: {len(customers_df):,}")
st.sidebar.markdown(f"- Transactions: {len(transactions_df):,}")
st.sidebar.markdown(f"- Alerts: {len(alerts_df):,}")
if bsi_df is not None:
    st.sidebar.markdown(f"- BSI Computed: {len(bsi_df):,}")


# ══════════════════════════════════════════════════════════════════════
# PAGE: ALERT DASHBOARD
# ══════════════════════════════════════════════════════════════════════
if page == "Alert Dashboard":
    st.title("AML Alert Monitoring Dashboard")
    st.caption("Real-time monitoring of suspicious activity across 10 typologies")

    # Merge risk scores
    alerts_scored = alerts_df.copy()
    if 'risk_score' not in alerts_scored.columns:
        alerts_scored = alerts_scored.merge(
            features_df[['customer_id', 'risk_score']].drop_duplicates(),
            on='customer_id', how='left'
        )

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    sar_files = glob.glob(os.path.join(OUTPUTS_DIR, "sar_alert_*.json"))
    c1.metric("Total Alerts", f"{len(alerts_df):,}")
    c2.metric("High Severity", f"{len(alerts_df[alerts_df['severity'] == 'high']):,}")
    c3.metric("Pending Review", f"{len(alerts_df[alerts_df['status'] == 'open']):,}")
    c4.metric("SARs Generated", f"{len(sar_files):,}")
    c5.metric("Suspicious Volume", f"${alerts_df['total_amount'].sum():,.0f}")

    # BSI summary if available
    if bsi_df is not None:
        st.markdown("---")
        st.subheader("Behavioral Stability Overview")
        bc1, bc2, bc3, bc4 = st.columns(4)
        drift_counts = bsi_df['drift_level'].value_counts()
        bc1.metric("Critical Drift", drift_counts.get('critical', 0))
        bc2.metric("High Drift", drift_counts.get('high', 0))
        bc3.metric("Moderate Drift", drift_counts.get('moderate', 0))
        bc4.metric("Stable", drift_counts.get('stable', 0))

    st.markdown("---")

    # Charts
    ch1, ch2 = st.columns(2)

    with ch1:
        st.subheader("Risk Score Distribution")
        if 'risk_score' in alerts_scored.columns:
            fig = px.histogram(alerts_scored, x='risk_score', nbins=20,
                               color_discrete_sequence=['#ff4b4b'])
            fig.add_vline(x=HIGH_RISK_THRESHOLD, line_dash="dash", line_color="red",
                          annotation_text="Threshold")
            fig.update_layout(height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.subheader("Alerts by Typology")
        tc = alerts_df['alert_type'].value_counts()
        fig = px.bar(x=tc.index, y=tc.values, color=tc.values,
                     color_continuous_scale='Reds',
                     labels={'x': 'Typology', 'y': 'Count'})
        fig.update_layout(height=350, showlegend=False, margin=dict(t=30, b=30))
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # BSI Distribution
    if bsi_df is not None:
        st.subheader("BSI Score Distribution (All Customers)")
        fig = px.histogram(bsi_df, x='bsi_score', nbins=30,
                           color_discrete_sequence=['#3b82f6'])
        fig.add_vline(x=25, line_dash="dash", line_color="red", annotation_text="Critical")
        fig.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="High Drift")
        fig.add_vline(x=70, line_dash="dash", line_color="yellow", annotation_text="Moderate")
        fig.update_layout(height=300, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Top alerts table
    st.subheader("Top 20 High-Risk Alerts")
    top = alerts_scored.nlargest(20, 'risk_score') if 'risk_score' in alerts_scored.columns else alerts_scored.head(20)
    top = top.merge(customers_df[['customer_id', 'name']], on='customer_id', how='left')

    if bsi_df is not None:
        top = top.merge(bsi_df[['customer_id', 'bsi_score', 'drift_level']], on='customer_id', how='left')

    display_cols = ['alert_id', 'name', 'alert_type', 'total_amount']
    col_names = ['Alert ID', 'Customer', 'Typology', 'Amount ($)']
    if 'risk_score' in top.columns:
        top['risk_pct'] = (top['risk_score'] * 100).round(1)
        display_cols.append('risk_pct')
        col_names.append('Risk %')
    if 'bsi_score' in top.columns:
        display_cols.extend(['bsi_score', 'drift_level'])
        col_names.extend(['BSI', 'Drift'])
    display_cols.append('status')
    col_names.append('Status')

    disp = top[display_cols].copy()
    disp.columns = col_names
    st.dataframe(disp, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: BEHAVIORAL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════
elif page == "Behavioral Intelligence":
    st.title("Behavioral Intelligence Dashboard")
    st.caption("Signal visualization, BSI timeline, network graph, and adaptive monitoring")

    alert_id = st.selectbox("Select Alert",
                            alerts_df['alert_id'].tolist(),
                            format_func=lambda x: f"Alert {x} - {alerts_df[alerts_df['alert_id']==x].iloc[0]['alert_type']}")

    alert_row = alerts_df[alerts_df['alert_id'] == alert_id].iloc[0]
    cid = alert_row['customer_id']
    cust = customers_df[customers_df['customer_id'] == cid].iloc[0]

    st.markdown(f"**Customer:** {cust['name']} | **Typology:** {alert_row['alert_type']} | "
                f"**Amount:** ${alert_row['total_amount']:,.2f}")

    # BSI Card
    if bsi_df is not None:
        bsi_row = bsi_df[bsi_df['customer_id'] == cid]
        if len(bsi_row) > 0:
            bsi = bsi_row.iloc[0]
            st.markdown("---")
            st.subheader("Behavioral Stability Index (BSI)")

            m1, m2, m3 = st.columns(3)
            m1.metric("BSI Score", f"{bsi['bsi_score']:.1f} / 100")
            m2.metric("Drift Level", bsi['drift_level'].title())

            if monitor_df is not None:
                mon = monitor_df[monitor_df['customer_id'] == cid]
                if len(mon) > 0:
                    m3.metric("Monitoring", mon.iloc[0]['monitoring_level'].title())

            # Radar + BSI Timeline side by side
            radar_col, timeline_col = st.columns(2)

            with radar_col:
                categories = ['Entropy', 'Temporal', 'Counterparty', 'Amount', 'Network']
                values = [
                    bsi.get('entropy_stability', 50),
                    bsi.get('temporal_stability', 50),
                    bsi.get('counterparty_stability', 50),
                    bsi.get('amount_stability', 50),
                    bsi.get('network_stability', 50),
                ]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name='Stability',
                    fillcolor='rgba(59, 130, 246, 0.3)', line_color='#3b82f6',
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=380, margin=dict(t=30, b=30), showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with timeline_col:
                st.markdown("**BSI Timeline** (behavioral regime changes)")
                try:
                    from services.behavioral_engine.bsi_timeline import compute_bsi_timeline
                    timeline_df = compute_bsi_timeline(cid, transactions_df)
                    if len(timeline_df) > 1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=timeline_df['window_midpoint'], y=timeline_df['bsi_score'],
                            mode='lines+markers', name='BSI',
                            line=dict(color='#3b82f6', width=2),
                            marker=dict(size=6),
                        ))
                        fig.add_hline(y=25, line_dash="dash", line_color="red",
                                      annotation_text="Critical")
                        fig.add_hline(y=50, line_dash="dash", line_color="orange",
                                      annotation_text="High Drift")
                        fig.add_hline(y=70, line_dash="dash", line_color="yellow",
                                      annotation_text="Moderate")
                        fig.update_layout(height=380, margin=dict(t=30, b=30),
                                          yaxis_title="BSI Score", xaxis_title="Time Window")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient transaction history for timeline")
                except Exception as e:
                    st.warning(f"Timeline computation: {e}")

    # Transaction Network Graph
    st.markdown("---")
    st.subheader("Transaction Network Graph")
    try:
        from services.graph_engine.graph_visualization import create_customer_graph_figure
        graph_fig = create_customer_graph_figure(cid, transactions_df)
        if graph_fig:
            st.plotly_chart(graph_fig, use_container_width=True)
        else:
            st.info("No counterparty relationships to visualize")
    except Exception as e:
        st.warning(f"Graph visualization: {e}")

    # Behavioral Signals Detail
    st.markdown("---")
    st.subheader("Behavioral Signal Details")

    feat_row = features_df[features_df['customer_id'] == cid]
    if len(feat_row) > 0:
        feat = feat_row.iloc[0]

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown("**Entropy Signals**")
            st.write(f"Amount Drift: {feat.get('amount_entropy_drift', 0):.3f}")
            st.write(f"Timing Drift: {feat.get('timing_entropy_drift', 0):.3f}")
            st.write(f"CP Drift: {feat.get('counterparty_entropy_drift', 0):.3f}")

        with s2:
            st.markdown("**Burst Signals**")
            st.write(f"Burstiness: {feat.get('burstiness_index', 0):.3f}")
            st.write(f"Burst Events: {int(feat.get('burst_events', 0))}")
            st.write(f"Max Daily: {int(feat.get('max_daily_count', 0))}")

        with s3:
            st.markdown("**Counterparty**")
            st.write(f"Expansion Rate: {feat.get('counterparty_expansion_rate', 0):.2f}")
            st.write(f"Novelty Slope: {feat.get('counterparty_novelty_slope', 0):.3f}")
            st.write(f"Unique CPs: {int(feat.get('total_unique_counterparties', 0))}")

        with s4:
            st.markdown("**Amount & Network**")
            st.write(f"Benford Dev: {feat.get('benford_deviation', 0):.3f}")
            st.write(f"Structuring: {feat.get('structuring_score', 0):.3f}")
            st.write(f"Flow Velocity: {feat.get('flow_velocity', 0):.4f}")
            st.write(f"Layer Depth: {int(feat.get('layer_depth', 0))}")
            st.write(f"Funnel Ratio: {feat.get('funnel_ratio', 1):.2f}")

        # Signal bar chart
        st.subheader("Signal Strength Comparison")
        signal_names = ['Entropy Drift', 'Burstiness', 'CP Expansion', 'Benford Dev',
                        'Structuring', 'Flow Velocity', 'Funnel Ratio']
        signal_vals = [
            feat.get('amount_entropy_drift', 0),
            abs(feat.get('burstiness_index', 0)),
            feat.get('counterparty_expansion_rate', 0) / 10,
            feat.get('benford_deviation', 0) / 2,
            feat.get('structuring_score', 0),
            feat.get('flow_velocity', 0) * 10,
            (feat.get('funnel_ratio', 1) - 1) / 10,
        ]

        fig = px.bar(x=signal_names, y=signal_vals,
                     color=signal_vals, color_continuous_scale='RdYlGn_r',
                     labels={'x': 'Signal', 'y': 'Normalized Strength'})
        fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Adaptive Monitoring
    if monitor_df is not None:
        st.markdown("---")
        st.subheader("Adaptive Monitoring State")
        mon = monitor_df[monitor_df['customer_id'] == cid]
        if len(mon) > 0:
            m = mon.iloc[0]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Level", m['monitoring_level'].title())
            mc2.metric("Check Interval", f"{int(m['check_interval_days'])} days")
            mc3.metric("Anomaly Threshold", f"{m['anomaly_threshold']:.1f}")
            if m.get('escalation_triggered'):
                st.warning(f"Escalation triggered: {m.get('recommended_action', '')}")
            else:
                st.success(m.get('recommended_action', 'Standard monitoring'))


# ══════════════════════════════════════════════════════════════════════
# PAGE: GENERATE SAR
# ══════════════════════════════════════════════════════════════════════
elif page == "Generate SAR":
    st.title("SAR Narrative Generator")
    st.caption("AI-powered SAR generation with behavioral intelligence context")

    c1, c2 = st.columns([1, 2])
    with c1:
        alert_id = st.number_input("Select Alert ID", min_value=0,
                                   max_value=len(alerts_df) - 1, value=0)

    alert_row = alerts_df[alerts_df['alert_id'] == alert_id]
    if len(alert_row) > 0:
        ar = alert_row.iloc[0]
        cust = customers_df[customers_df['customer_id'] == ar['customer_id']]
        with c2:
            if len(cust) > 0:
                cr = cust.iloc[0]
                st.markdown(f"**{cr['name']}** | {ar['alert_type'].replace('_',' ').title()} | "
                            f"${ar['total_amount']:,.2f}")

    # Check LLM availability
    from services.sar.sar_fallback import is_ollama_available
    ollama_up = is_ollama_available()
    if not ollama_up:
        st.info("Ollama not detected. SAR will be generated using template-based fallback mode. "
                "Start Ollama with `ollama serve` for AI-powered narratives.")

    if st.button("Generate SAR Narrative", type="primary"):
        with st.spinner("Generating SAR narrative with behavioral intelligence..."):
            try:
                from services.sar.sar_generator import load_alert_data, save_sar
                from services.sar.compliance import validate_sar
                from services.sar.audit import create_audit_tables, save_sar_record

                create_audit_tables()
                alert, customer, txns = load_alert_data(alert_id)

                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.metric("Customer", customer['name'])
                ic2.metric("Typology", alert['alert_type'].replace('_', ' ').title())
                ic3.metric("Amount", f"${alert['total_amount']:,.2f}")
                ic4.metric("Transactions", f"{alert['num_transactions']}")

                # Get SHAP drivers
                from services.sar.sar_generator import get_shap_drivers
                top_drivers, risk_score = get_shap_drivers(alert, features_df, explainer, feature_cols)

                # BSI data
                bsi_data = None
                if bsi_df is not None:
                    bsi_row = bsi_df[bsi_df['customer_id'] == alert['customer_id']]
                    if len(bsi_row) > 0:
                        bsi_data = bsi_row.iloc[0].to_dict()

                if ollama_up:
                    # Full LLM generation
                    from langchain_community.vectorstores import Chroma
                    from langchain_ollama import OllamaEmbeddings, OllamaLLM
                    from services.sar.sar_generator import generate_narrative

                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1, num_predict=2000)

                    narrative, audit_trail = generate_narrative(
                        alert_id, vectorstore, llm, features_df, explainer, feature_cols, bsi_df
                    )
                else:
                    # Fallback: template-based generation
                    from services.sar.sar_fallback import generate_fallback_narrative
                    from datetime import datetime

                    narrative = generate_fallback_narrative(
                        alert, customer, txns, risk_score, bsi_data, top_drivers
                    )
                    audit_trail = {
                        'alert_id': int(alert_id),
                        'generation_timestamp': datetime.now().isoformat(),
                        'model': 'template-fallback (Ollama unavailable)',
                        'temperature': 0.0,
                        'customer_id': int(customer['customer_id']),
                        'customer_name': customer['name'],
                        'num_transactions_analyzed': len(txns),
                        'transaction_date_range': f"{txns['transaction_date'].min()} to {txns['transaction_date'].max()}",
                        'risk_score': float(risk_score),
                        'typology_detected': alert['alert_type'],
                        'shap_top_features': top_drivers.head(5).to_dict('records') if len(top_drivers) > 0 else [],
                        'templates_retrieved': [],
                        'generation_params': {'temperature': 0.0, 'max_tokens': 0},
                        'narrative': narrative,
                        'narrative_word_count': len(narrative.split()),
                        'bsi_score': bsi_data.get('bsi_score') if bsi_data else None,
                        'drift_level': bsi_data.get('drift_level') if bsi_data else None,
                        'status': 'draft',
                        'reviewed_by': None,
                        'approved_by': None,
                        'filed_date': None,
                    }

                compliance = validate_sar(narrative)
                save_sar(alert_id, narrative, audit_trail, compliance)
                save_sar_record(alert_id, int(customer['customer_id']),
                                narrative, audit_trail, compliance)

                # Display
                st.subheader("Generated SAR Narrative")
                if compliance['compliant']:
                    st.success(f"COMPLIANT - {compliance['sections_found']}/{compliance['sections_total']} sections | "
                               f"{compliance['word_count']} words")
                else:
                    issues = []
                    if compliance['missing_sections']:
                        issues.append(f"Missing: {', '.join(compliance['missing_sections'])}")
                    if not compliance['word_count_ok']:
                        issues.append(f"Word count: {compliance['word_count']}")
                    st.warning(f"Review needed - {' | '.join(issues)}")

                st.text_area("SAR Narrative (Editable)", value=narrative, height=500)

                # Compliance
                st.subheader("Compliance Checklist")
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.write("**Structural**")
                    st.write(f"- 5W+H Sections: {'PASS' if compliance['has_all_sections'] else 'FAIL'}")
                    st.write(f"- Word Count ({compliance['word_count']}): {'PASS' if compliance['word_count_ok'] else 'FAIL'}")
                with cc2:
                    st.write("**Content**")
                    st.write(f"- Dollar Amounts: {'PASS' if compliance['has_specific_amounts'] else 'FAIL'}")
                    st.write(f"- Dates: {'PASS' if compliance['has_specific_dates'] else 'FAIL'}")
                    st.write(f"- Behavioral Ref: {'PASS' if compliance['has_behavioral_reference'] else 'MISSING'}")

                # SHAP
                st.subheader("Risk Drivers (SHAP)")
                if audit_trail.get('shap_top_features'):
                    sdf = pd.DataFrame(audit_trail['shap_top_features'])
                    if len(sdf) > 0:
                        sdf['label'] = sdf['feature'].str.replace('_', ' ').str.title()
                        fig = px.bar(sdf, x='shap_value', y='label', orientation='h',
                                     color='shap_value', color_continuous_scale='RdYlGn_r')
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                # BSI context
                if audit_trail.get('bsi_score') is not None:
                    st.info(f"BSI Score: {audit_trail['bsi_score']:.1f}/100 "
                            f"(Drift: {audit_trail.get('drift_level', 'N/A')})")

                with st.expander("View Audit Trail"):
                    st.json({k: v for k, v in audit_trail.items() if k != 'narrative'})

                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button("Download Narrative (.txt)", narrative,
                                       f"sar_narrative_{alert_id}.txt", "text/plain")
                with dc2:
                    doc = json.dumps({'narrative': narrative, 'audit_trail': audit_trail,
                                      'compliance': compliance}, indent=2, ensure_ascii=False)
                    st.download_button("Download Full SAR (.json)", doc,
                                       f"sar_complete_{alert_id}.json", "application/json")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════
# PAGE: INVESTIGATION
# ══════════════════════════════════════════════════════════════════════
elif page == "Investigation":
    st.title("AI Alert Investigation")
    st.caption("Context-aware AML investigation assistant. Scoped strictly to the selected alert.")

    # ── Alert selector ────────────────────────────────────────────────
    alert_id = st.selectbox(
        "Select Alert",
        alerts_df['alert_id'].tolist(),
        format_func=lambda x: (
            f"Alert {x} — "
            f"{alerts_df[alerts_df['alert_id']==x].iloc[0]['alert_type'].replace('_',' ').title()} | "
            f"{alerts_df[alerts_df['alert_id']==x].iloc[0]['severity'].title()} | "
            f"${alerts_df[alerts_df['alert_id']==x].iloc[0]['total_amount']:,.0f}"
        ),
        key="inv_alert_select",
    )

    ar = alerts_df[alerts_df['alert_id'] == alert_id].iloc[0]
    cust_row = customers_df[customers_df['customer_id'] == ar['customer_id']]
    if len(cust_row) > 0:
        cr = cust_row.iloc[0]
        st.markdown(
            f"**{cr['name']}** &nbsp;|&nbsp; {cr['occupation']} &nbsp;|&nbsp; "
            f"Risk: **{cr['risk_rating'].title()}** &nbsp;|&nbsp; "
            f"Volume: **${ar['total_amount']:,.2f}**",
            unsafe_allow_html=True,
        )

    # ── LLM availability check ────────────────────────────────────────
    from services.sar.sar_fallback import is_ollama_available
    if not is_ollama_available():
        st.error("Ollama not detected. Start Ollama with `ollama serve` to use the Investigation Chatbot.")
        st.stop()

    st.markdown("---")

    # ── Alert-scoped session state ────────────────────────────────────
    # Each alert_id gets its own isolated session object.
    # Switching alerts preserves prior conversations — analyst can navigate back.
    # system_prompt is built once per alert and cached here (expensive: DB + SHAP).
    if "alert_sessions" not in st.session_state:
        st.session_state.alert_sessions = {}

    if alert_id not in st.session_state.alert_sessions:
        st.session_state.alert_sessions[alert_id] = {
            "messages": [],
            "system_prompt": None,   # built lazily on first send
        }

    session = st.session_state.alert_sessions[alert_id]

    # ── Download transactions button ──────────────────────────────────
    from services.investigation_tools import get_alert_transactions
    with st.expander("Download transactions for this alert"):
        if st.button("Prepare CSV", key=f"dl_prep_{alert_id}"):
            txn_export = get_alert_transactions(alert_id, limit=10000)
            csv_bytes = txn_export.to_csv(index=False).encode()
            st.download_button(
                label=f"Download transactions_{alert_id}.csv",
                data=csv_bytes,
                file_name=f"transactions_{alert_id}.csv",
                mime="text/csv",
                key=f"dl_btn_{alert_id}",
            )

    st.markdown("---")

    # ── Chat history display ──────────────────────────────────────────
    for message in session["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── User input ────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about this alert..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        session["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            from langchain_ollama import OllamaLLM
            from services.chatbot import build_system_prompt, stream_investigation_response

            # Build system prompt once per alert (cached in session state)
            if session["system_prompt"] is None:
                with st.spinner("Loading alert context..."):
                    # Pull counterfactual if already computed (from Counterfactual page cache)
                    cf_data = st.session_state.get(f"cf_cache_{alert_id}")
                    session["system_prompt"] = build_system_prompt(
                        alert_id, features_df, explainer, feature_cols, bsi_df,
                        counterfactual=cf_data,
                    )

            llm = OllamaLLM(model=LLM_MODEL, temperature=0.0, num_predict=2500,
                            keep_alive=-1, repeat_penalty=1.3)

            stream = stream_investigation_response(
                llm,
                session["system_prompt"],
                prompt,
                session["messages"][:-1],   # history excludes current user turn
            )
            response = st.write_stream(stream)

        session["messages"].append({"role": "assistant", "content": response})


# ══════════════════════════════════════════════════════════════════════
# PAGE: COUNTERFACTUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "Counterfactual Analysis":
    st.title("Counterfactual SAR Simulation")
    st.caption("What would need to change to flip the SAR decision?")

    alert_id = st.selectbox(
        "Select Alert",
        alerts_df['alert_id'].tolist(),
        format_func=lambda x: f"Alert {x} — {alerts_df[alerts_df['alert_id']==x].iloc[0]['alert_type'].replace('_',' ').title()}",
        key="cf_alert_select",
    )

    alert_row = alerts_df[alerts_df['alert_id'] == alert_id].iloc[0]
    cid = alert_row['customer_id']

    # Auto-compute counterfactual — cached per alert_id in session state so it
    # runs once per alert, not on every Streamlit rerun.
    # Result is also stored in st.session_state under cf_cache_{alert_id} so the
    # Investigation chatbot can pick it up as Tier-1 context without recomputing.
    @st.cache_data(show_spinner=False)
    def _compute_cf(customer_id, _features_df, _feature_cols):
        from services.sar.counterfactual import generate_counterfactual
        _model = joblib.load(os.path.join(MODELS_DIR, 'risk_classifier.pkl'))
        return generate_counterfactual(customer_id, _features_df, _model, _feature_cols)

    with st.spinner("Computing counterfactual analysis..."):
        cf = _compute_cf(cid, features_df, feature_cols)

    # Share result with Investigation chatbot via session state
    st.session_state[f"cf_cache_{alert_id}"] = cf

    if cf is None:
        st.error("No feature data available for this alert.")
    else:
        # Summary metrics
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Current Risk", f"{cf['current_risk_score']:.1%}")
        sc2.metric("SAR Triggered", "YES" if cf['sar_triggered'] else "NO")
        sc3.metric("Barely Crossed", "YES" if cf['summary']['barely_crossed_threshold'] else "NO")

        st.markdown("---")

        # Dimension Impact chart
        st.subheader("Dimension Impact Analysis")

        dims = cf['dimension_impacts']
        dim_names    = [d['dimension'] for d in dims]
        contributions = [d['contribution_pct'] for d in dims]
        flips        = [d['would_flip_decision'] for d in dims]
        colors       = ['#ef4444' if f else '#3b82f6' for f in flips]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=dim_names[::-1],
            x=contributions[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{c:.1f}%" for c in contributions[::-1]],
            textposition='outside',
        ))
        fig.update_layout(
            height=400,
            xaxis_title="Risk Contribution (%)",
            margin=dict(t=10, b=10, l=200),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Dimension details
        st.subheader("Detailed Counterfactuals")
        for d in dims:
            status = "🔴 WOULD FLIP DECISION" if d['would_flip_decision'] else ""
            with st.expander(f"{d['dimension']} ({d['contribution_pct']:.1f}% of risk) {status}"):
                st.write(f"**Risk if normalized:** {d['risk_with_normalization']:.2%} "
                         f"(reduction: {d['risk_reduction']:.2%})")
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.write("**Current Values**")
                    for feat, val in d['current_values'].items():
                        st.write(f"- {feat}: {val:.4f}")
                with dc2:
                    st.write("**Counterfactual (Normal)**")
                    for feat, val in d['counterfactual_values'].items():
                        st.write(f"- {feat}: {val:.4f}")

        st.markdown("---")
        st.info(
            f"**Top risk drivers:** {', '.join(cf['summary']['top_risk_dimensions'])} — "
            f"these explain {cf['summary']['top_2_contribution_pct']:.0f}% of total risk."
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE: AUDIT TRAIL VIEWER
# ══════════════════════════════════════════════════════════════════════
elif page == "Audit Trail Viewer":
    st.title("SAR Audit Trail Viewer")
    st.caption("Complete transparency: data lineage, model decisions, compliance validation")

    sar_files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "sar_alert_*.json")))

    if not sar_files:
        st.info("No SARs generated yet. Go to 'Generate SAR' to create one.")
    else:
        options = {}
        for f in sar_files:
            num = os.path.basename(f).replace('sar_alert_', '').replace('.json', '')
            options[f"Alert {num}"] = f

        selected = st.selectbox("Select SAR", list(options.keys()))

        if selected:
            with open(options[selected], 'r', encoding='utf-8') as f:
                sar_doc = json.load(f)

            audit = sar_doc.get('audit_trail', {})
            compliance = sar_doc.get('compliance_check', {})

            # Metadata
            st.subheader("SAR Metadata")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Alert ID", audit.get('alert_id', 'N/A'))
            mc2.metric("Risk Score", f"{audit.get('risk_score', 0):.1%}")
            mc3.metric("BSI Score", f"{audit.get('bsi_score', 'N/A')}")
            mc4.metric("Words", audit.get('narrative_word_count', 'N/A'))
            mc5.metric("Status", audit.get('status', 'N/A').title())

            st.markdown("---")

            lc, rc = st.columns(2)

            with lc:
                st.subheader("Data Lineage")
                st.write(f"**Customer:** {audit.get('customer_name', 'N/A')} (ID: {audit.get('customer_id', 'N/A')})")
                st.write(f"**Transactions:** {audit.get('num_transactions_analyzed', 'N/A')}")
                st.write(f"**Date Range:** {audit.get('transaction_date_range', 'N/A')}")

                st.subheader("Model Decision")
                st.write(f"**Typology:** {audit.get('typology_detected', 'N/A')}")
                st.write(f"**Risk Score:** {audit.get('risk_score', 0):.2%}")
                st.write(f"**BSI:** {audit.get('bsi_score', 'N/A')}")
                st.write(f"**Drift Level:** {audit.get('drift_level', 'N/A')}")

            with rc:
                st.subheader("Template Retrieval (RAG)")
                for i, t in enumerate(audit.get('templates_retrieved', [])):
                    st.write(f"Template {i+1}: {t.get('typology', '?')} (ID: {t.get('template_id', '?')})")

                st.subheader("Top SHAP Features")
                for feat in audit.get('shap_top_features', [])[:5]:
                    name = feat.get('feature', '').replace('_', ' ').title()
                    st.write(f"- **{name}:** {feat.get('shap_value', 0):+.3f}")

                st.subheader("LLM Generation")
                st.write(f"**Timestamp:** {audit.get('generation_timestamp', 'N/A')}")
                gp = audit.get('generation_params', {})
                st.write(f"**Temperature:** {gp.get('temperature', 'N/A')}")
                st.write(f"**Model:** {audit.get('model', 'N/A')}")

            st.markdown("---")

            # Compliance
            st.subheader("Compliance Validation")
            if compliance.get('compliant'):
                st.success(f"COMPLIANT - {compliance.get('sections_found', 0)}/{compliance.get('sections_total', 8)} sections")
            else:
                missing = compliance.get('missing_sections', [])
                st.warning(f"Review required - Missing: {', '.join(missing) if missing else 'None'}")

            vc1, vc2 = st.columns(2)
            with vc1:
                st.write(f"- All sections: {'PASS' if compliance.get('has_all_sections') else 'FAIL'}")
                st.write(f"- Word count: {'PASS' if compliance.get('word_count_ok') else 'FAIL'}")
            with vc2:
                st.write(f"- Dollar amounts: {'PASS' if compliance.get('has_specific_amounts') else 'FAIL'}")
                st.write(f"- Dates: {'PASS' if compliance.get('has_specific_dates') else 'FAIL'}")
                st.write(f"- Behavioral ref: {'PASS' if compliance.get('has_behavioral_reference') else 'MISSING'}")

            st.markdown("---")

            st.subheader("Narrative")
            st.text_area("SAR Narrative", value=sar_doc.get('narrative', ''), height=500, disabled=True)

            with st.expander("Raw Audit Trail JSON"):
                st.json({k: v for k, v in audit.items() if k != 'narrative'})
