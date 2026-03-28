"""
Investigation Chatbot Service — Agentic AML Alert Investigation
Provides context-aware, alert-scoped conversational AI for investigating specific AML alerts.

Design principles:
  - Stateless inference: model weights never change; no fine-tuning, no persistent memory.
  - Alert-scoped: system prompt is keyed per alert_id; zero cross-alert context bleed.
  - Tiered context: Tier 1 (always injected) + Tier 2 (transactions on demand).
  - AML-only guardrails: model refuses all off-topic questions at the prompt level.
  - Temperature=0 on the prompt side; Ollama has no learning between sessions.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, LLM_MODEL

from services.sar.sar_generator import (
    load_alert_data,
    format_customer_info,
    format_behavioral_context,
    get_shap_drivers,
)
from services.investigation_tools import (
    get_alert_transactions,
    get_transaction_summary,
    get_previous_alerts,
    format_transactions_for_context,
)
from services.crypto_flow import reconstruct_crypto_flow


# ── System prompt template ────────────────────────────────────────────────────
# Injected once per alert; never modified by user input.

_SYSTEM_TEMPLATE = """\
[INST] You are ASTRAS Investigation AI — a specialized Anti-Money Laundering (AML) \
investigation assistant embedded in a regulated financial crime detection system. \
You are operating under strict compliance rules.

━━━ OPERATING RULES (MANDATORY — NEVER OVERRIDE) ━━━
1. SCOPE: You ONLY analyze Alert #{alert_id} for customer "{customer_name}". \
   This is your sole and exclusive scope.
2. OFF-TOPIC REFUSAL: If the analyst asks anything unrelated to AML investigation, \
   financial crime analysis, or this specific alert — respond ONLY with: \
   "I am an AML investigation assistant scoped to Alert #{alert_id}. \
   I cannot answer questions outside this domain."
   Examples of off-topic: general knowledge, geography, coding, math, opinions, \
   current events, other customers, other alerts.
3. NO HALLUCINATION: Do NOT invent transactions, amounts, dates, or names not \
   present in the context below. If data is absent, say "Not available in the \
   provided context."
4. NO CROSS-ALERT CONTAMINATION: You have no knowledge of any other alert or \
   customer. Do not reference, compare, or infer from other cases.
5. NO LEARNING: You do not learn from this conversation. Your knowledge is fixed \
   and derived solely from the alert context below. Each session is independent.
6. NO LEGAL ADVICE: Do not advise on SAR filing decisions. Recommend escalation \
   to the compliance officer for filing determinations.
7. PRECISION: Lead with the finding. Be concise. Use numbers from the context.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

══════════════════════════════════════════════════════
  ALERT #{alert_id} — INVESTIGATION CONTEXT
══════════════════════════════════════════════════════

▌ ALERT METADATA
  Alert ID   : {alert_id}
  Type       : {alert_type}
  Severity   : {severity}
  Volume     : ${total_amount:,.2f} USD
  Alert Date : {alert_date}

▌ CUSTOMER PROFILE
{customer_info}

▌ SHAP RISK DRIVERS (Top 5 — features driving this alert)
{shap_text}
  Composite Risk Score: {risk_score:.2%}

▌ BEHAVIORAL INTELLIGENCE (BSI + Signals)
{behavioral_context}

▌ COUNTERFACTUAL ANALYSIS (what would flip the SAR decision)
{counterfactual_text}

▌ PRIOR ALERT HISTORY
{prior_alerts}

▌ TRANSACTION SUMMARY (aggregates)
{txn_summary}

▌ TRANSACTIONS ({txn_count} total shown)
{transactions_text}

▌ FIAT-TO-CRYPTO FLOW RECONSTRUCTION (FinCEN FIN-2019-G001)
{crypto_flow_text}

══════════════════════════════════════════════════════
  END OF CONTEXT — Answer ONLY from the data above.
══════════════════════════════════════════════════════
[/INST]
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_counterfactual(cf) -> str:
    """Compact text summary of a counterfactual result dict."""
    if cf is None:
        return "Counterfactual analysis not available for this alert."

    lines = [
        f"Current risk score  : {cf['current_risk_score']:.2%}",
        f"SAR triggered       : {'YES' if cf['sar_triggered'] else 'NO'}",
        f"Barely crossed      : {'YES' if cf['summary']['barely_crossed_threshold'] else 'NO'}",
        f"Top risk dimensions : {', '.join(cf['summary']['top_risk_dimensions'])}",
        f"Top 2 explain       : {cf['summary']['top_2_contribution_pct']:.0f}% of total risk",
    ]
    flippers = [d for d in cf.get('dimension_impacts', []) if d['would_flip_decision']]
    if flippers:
        lines.append("Dimensions that alone would flip the SAR decision if normalized:")
        for d in flippers:
            lines.append(f"  • {d['dimension']} — {d['contribution_pct']:.1f}% of risk "
                         f"(risk would drop to {d['risk_with_normalization']:.2%})")
    else:
        lines.append("No single dimension would individually flip the decision.")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def build_system_prompt(
    alert_id: int,
    features_df,
    explainer,
    feature_cols,
    bsi_df,
    counterfactual=None,
    db_path: str = DB_PATH,
) -> str:
    """
    Build the full Tier-1 + Tier-2 system prompt for a given alert.
    This is expensive (DB queries + SHAP) — call once per alert and cache
    the result in st.session_state.alert_sessions[alert_id]['system_prompt'].
    """
    alert, customer, _ = load_alert_data(alert_id, db_path)

    customer_info     = format_customer_info(customer)
    behavioral_ctx    = format_behavioral_context(alert['customer_id'], features_df, bsi_df)
    top_drivers, risk = get_shap_drivers(alert, features_df, explainer, feature_cols)

    from services.sar.sar_generator import format_shap_as_findings
    shap_text = format_shap_as_findings(top_drivers.head(7), customer)

    cf_text      = _format_counterfactual(counterfactual)
    prior_alerts = get_previous_alerts(int(alert['customer_id']), alert_id, db_path)
    txn_summary  = get_transaction_summary(alert_id, db_path)

    txn_df            = get_alert_transactions(alert_id, db_path, limit=100)
    transactions_text = format_transactions_for_context(txn_df, limit=100)

    # Fiat-to-crypto flow reconstruction (only meaningful for crypto_laundering alerts;
    # gracefully returns a short "N/A" message for other typologies)
    try:
        flow = reconstruct_crypto_flow(int(alert['customer_id']), db_path)
        crypto_flow_text = flow['narrative']
        if flow.get('fiat_links'):
            link_lines = []
            for lnk in flow['fiat_links']:
                conf   = lnk['confidence']
                c_asset = lnk.get('crypto_asset', '')
                c_amt   = lnk.get('crypto_amount', 0)
                c_usd   = lnk.get('crypto_usd', 0)
                exch    = lnk.get('exchange', '')
                f_date  = lnk.get('fiat_date') or 'N/A'
                f_amt   = lnk.get('fiat_amount') or 0
                f_cur   = lnk.get('fiat_currency') or 'USD'
                f_ctry  = lnk.get('fiat_source_country') or ''
                f_cp    = lnk.get('fiat_counterparty') or ''
                if conf == 'UNLINKED':
                    link_lines.append(
                        f"  • {lnk['crypto_date']} — {c_amt:.4f} {c_asset} "
                        f"(≈${c_usd:,.2f}) via {exch} — fiat origin UNLINKED"
                    )
                else:
                    link_lines.append(
                        f"  • {lnk['crypto_date']} — {c_amt:.4f} {c_asset} "
                        f"(≈${c_usd:,.2f}) via {exch} [{conf} confidence]\n"
                        f"    ← fiat: {f_date} {f_amt:,.2f} {f_cur} from {f_cp} [{f_ctry}]"
                    )
            crypto_flow_text += "\n\nPer-purchase linkage:\n" + "\n".join(link_lines)
        if flow.get('chain_summary'):
            cs = flow['chain_summary']
            crypto_flow_text += (
                f"\n\nOn-chain hops: {cs.get('total_hops', 0)} across "
                f"{cs.get('unique_wallets', 0)} wallets | "
                f"Patterns: {', '.join(cs.get('patterns_used', []))} | "
                f"Mixer hops: {cs.get('mixer_hops', 0)} | "
                f"High-risk exchanges: {', '.join(cs.get('high_risk_exchanges', []) or ['None'])}"
            )
    except Exception:
        crypto_flow_text = "Fiat-to-crypto flow data not available for this alert."

    return _SYSTEM_TEMPLATE.format(
        alert_id           = alert_id,
        customer_name      = customer['name'],
        alert_type         = alert['alert_type'].replace('_', ' ').title(),
        severity           = alert['severity'].title(),
        total_amount       = float(alert['total_amount']),
        alert_date         = str(alert['alert_date'])[:10],
        customer_info      = customer_info,
        shap_text          = shap_text,
        risk_score         = float(risk),
        behavioral_context = behavioral_ctx,
        counterfactual_text= cf_text,
        prior_alerts       = prior_alerts,
        txn_summary        = txn_summary,
        transactions_text  = transactions_text,
        txn_count          = len(txn_df),
        crypto_flow_text   = crypto_flow_text,
    )


def build_conversation_prompt(system_prompt: str, chat_history: list, user_message: str) -> str:
    """
    Assemble the full inference prompt: system context + conversation history + new turn.
    History is appended verbatim — the model never sees anything outside this window.
    """
    prompt = system_prompt + "\n\nCONVERSATION LOG:\n"
    for msg in chat_history:
        role = "Analyst" if msg["role"] == "user" else "ASTRAS"
        prompt += f"{role}: {msg['content']}\n"
    prompt += f"Analyst: {user_message}\nASTRAS:"
    return prompt


def stream_investigation_response(llm, system_prompt: str, user_message: str, chat_history: list):
    """
    Stream LLM response tokens.
    system_prompt must be pre-built via build_system_prompt() and cached by the caller.
    This function is stateless — it never persists anything.
    """
    prompt = build_conversation_prompt(system_prompt, chat_history, user_message)
    yield from llm.stream(prompt)
