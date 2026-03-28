"""
Investigation Agent
Agentic AML alert investigation using LangGraph ReAct + ChatOllama tool calling.

Architecture:
  - LangGraph create_react_agent (ReAct loop: think → tool call → observe → answer)
  - ChatOllama with .bind_tools() for native Mistral function-calling format
  - 5 alert-scoped read-only tools (no raw SQL, no cross-alert access)
  - System message carries the same AML guardrails + Tier-1 context as the
    static chatbot, but tools replace the need to cram everything into the prompt

Tools:
  filter_transactions  — parameterized query (country/method/type/amount)
  get_crypto_flow      — full fiat→crypto reconstruction + on-chain hops
  get_counterparty     — all transactions with a named entity
  get_transaction_by_id— single transaction detail lookup
  get_timeline         — date-range slice of transactions
"""
import os
import sys
import sqlite3

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from config import DB_PATH, LLM_MODEL


# ── Shared DB helper ──────────────────────────────────────────────────────────

def _query_customer_txns(customer_id: int, db_path: str,
                          country=None, method=None, txn_type=None,
                          min_amount=None, max_amount=None,
                          start_date=None, end_date=None,
                          counterparty=None, limit=200) -> pd.DataFrame:
    """Parameterised read-only query — no raw SQL exposed to LLM."""
    conn   = sqlite3.connect(db_path)
    sql    = """SELECT transaction_id, transaction_date, transaction_type,
                       amount, currency, usd_rate, method, location,
                       country, counterparty, counterparty_bank,
                       crypto_asset, description
                FROM transactions WHERE customer_id = ?"""
    params = [customer_id]

    if country:       sql += " AND LOWER(country)       = LOWER(?)";  params.append(country)
    if method:        sql += " AND LOWER(method)        = LOWER(?)";  params.append(method)
    if txn_type:      sql += " AND LOWER(transaction_type) = LOWER(?)"; params.append(txn_type)
    if min_amount:    sql += " AND amount >= ?";                        params.append(min_amount)
    if max_amount:    sql += " AND amount <= ?";                        params.append(max_amount)
    if start_date:    sql += " AND transaction_date >= ?";              params.append(start_date)
    if end_date:      sql += " AND transaction_date <= ?";              params.append(end_date)
    if counterparty:  sql += " AND LOWER(counterparty) LIKE LOWER(?)"; params.append(f"%{counterparty}%")

    sql += " ORDER BY transaction_date DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    if not df.empty and 'usd_rate' in df.columns:
        df['usd_amount'] = (df['amount'] * df['usd_rate'].fillna(1.0)).round(2)
    return df


def _df_to_text(df: pd.DataFrame, max_rows=30) -> str:
    if df.empty:
        return "No transactions match the specified filters."
    rows = []
    for _, r in df.head(max_rows).iterrows():
        cur = r.get('currency', 'USD')
        amt = r['amount']
        usd = r.get('usd_amount', amt)
        amt_str = (f"{amt:,.6f} {cur} (≈${usd:,.2f} USD)"
                   if cur != 'USD' else f"${amt:,.2f}")
        line = (f"- {str(r['transaction_date'])[:10]}  "
                f"{r['transaction_type'].title():<12} {amt_str:<35} "
                f"via {r['method']}")
        if pd.notna(r.get('counterparty')) and str(r['counterparty']) not in ('None', '', 'nan'):
            line += f"  |  {r['counterparty']}"
        if r.get('country') and r['country'] != 'USA':
            line += f"  [{r['country']}]"
        if pd.notna(r.get('crypto_asset')) and str(r['crypto_asset']) not in ('None', '', 'nan'):
            line += f"  ({r['crypto_asset']})"
        rows.append(line)
    result = "\n".join(rows)
    if len(df) > max_rows:
        result += f"\n... {len(df) - max_rows} more rows not shown."
    return result


# ── Tool factory — closures keep every tool scoped to one alert ───────────────

def _make_tools(alert_id: int, customer_id: int, db_path: str):

    @tool
    def filter_transactions(
        country: str = None,
        method: str = None,
        txn_type: str = None,
        min_amount: float = None,
        max_amount: float = None,
    ) -> str:
        """
        Retrieve transactions for this alert. Call with NO arguments to get ALL transactions.
        Optionally filter by:
          country   — e.g. "India", "UAE", "Germany"
          method    — "wire", "crypto_exchange", "ach", "card", "check", "cash"
          txn_type  — "deposit" or "withdrawal"
          min_amount / max_amount — native currency amount bounds
        To list ALL transactions: filter_transactions()
        """
        try:
            df = _query_customer_txns(
                customer_id, db_path,
                country=country, method=method, txn_type=txn_type,
                min_amount=min_amount, max_amount=max_amount,
            )
            total = len(df)
            usd_total = df['usd_amount'].sum() if 'usd_amount' in df.columns else df['amount'].sum()
            header = f"{total} transaction(s) found — total ≈${usd_total:,.2f} USD\n"
            return header + _df_to_text(df)
        except Exception as e:
            return f"Error querying transactions: {e}"

    @tool
    def get_transaction_by_id(transaction_id: int) -> str:
        """
        Return full detail for a single transaction by its transaction_id integer.
        Useful when you need to examine one specific transaction closely.
        """
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(
                """SELECT * FROM transactions
                   WHERE transaction_id = ? AND customer_id = ?""",
                conn, params=(transaction_id, customer_id)
            )
            conn.close()
            if df.empty:
                return f"Transaction {transaction_id} not found for this alert's customer."
            r = df.iloc[0]
            lines = [f"Transaction ID : {r['transaction_id']}",
                     f"Date           : {r['transaction_date']}",
                     f"Type           : {r['transaction_type']}",
                     f"Amount         : {r['amount']} {r.get('currency','USD')}",
                     f"USD Rate       : {r.get('usd_rate', 1.0)}",
                     f"USD Equivalent : ${float(r['amount']) * float(r.get('usd_rate',1.0)):,.2f}",
                     f"Method         : {r['method']}",
                     f"Location       : {r.get('location','')}",
                     f"Country        : {r.get('country','')}",
                     f"Counterparty   : {r.get('counterparty','')}",
                     f"Bank           : {r.get('counterparty_bank','')}",
                     f"Crypto Asset   : {r.get('crypto_asset','')}",
                     f"Description    : {r.get('description','')}"]
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_counterparty(name: str) -> str:
        """
        Return all transactions involving a specific counterparty name (partial match).
        Use this to trace a single entity's activity across the full transaction history.
        Example: get_counterparty("Richardson Inc") or get_counterparty("BitVault")
        """
        try:
            df = _query_customer_txns(customer_id, db_path, counterparty=name, limit=100)
            if df.empty:
                return f"No transactions found involving counterparty matching '{name}'."
            total_usd = df['usd_amount'].sum() if 'usd_amount' in df.columns else df['amount'].sum()
            types     = df['transaction_type'].value_counts().to_dict()
            header = (f"Counterparty '{name}': {len(df)} transaction(s), "
                      f"≈${total_usd:,.2f} USD total | {types}\n")
            return header + _df_to_text(df)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_timeline(start_date: str = None, end_date: str = None) -> str:
        """
        Return transactions within a date range (YYYY-MM-DD format).
        Leave either parameter blank to query from the beginning or to the end.
        Example: get_timeline("2025-02-01", "2025-04-30")
        """
        try:
            df = _query_customer_txns(
                customer_id, db_path,
                start_date=start_date, end_date=end_date, limit=200
            )
            df = df.sort_values('transaction_date')
            if df.empty:
                return f"No transactions found between {start_date} and {end_date}."
            total_usd = df['usd_amount'].sum() if 'usd_amount' in df.columns else df['amount'].sum()
            header = (f"Timeline {start_date or 'start'} → {end_date or 'end'}: "
                      f"{len(df)} transactions, ≈${total_usd:,.2f} USD\n")
            return header + _df_to_text(df)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_crypto_flow() -> str:
        """
        Return the complete fiat-to-cryptocurrency flow reconstruction for this customer.
        Includes:
          - Which fiat deposits preceded each crypto purchase (confidence-scored linkage)
          - On-chain hop analysis (mixer nodes, wallet hops, exchange KYC arbitrage)
          - SOURCE OF FIAT FUNDS narrative per FinCEN FIN-2019-G001 guidance
        Call this whenever the analyst asks about source of funds, crypto trail,
        on-chain activity, blockchain hops, or mixer usage.
        """
        try:
            from services.crypto_flow import reconstruct_crypto_flow
            flow = reconstruct_crypto_flow(customer_id, db_path)

            parts = [flow['narrative']]

            if flow.get('fiat_links'):
                parts.append("\nPer-purchase linkage detail:")
                for lnk in flow['fiat_links']:
                    conf  = lnk['confidence']
                    asset = lnk.get('crypto_asset', '')
                    c_amt = lnk.get('crypto_amount', 0)
                    c_usd = lnk.get('crypto_usd', 0)
                    exch  = lnk.get('exchange', '')
                    if conf == 'UNLINKED':
                        parts.append(
                            f"  • {lnk['crypto_date']} — {c_amt:.4f} {asset} "
                            f"(≈${c_usd:,.2f} USD) via {exch} — UNLINKED (no fiat deposit "
                            f"found within {10}-day window)"
                        )
                    else:
                        parts.append(
                            f"  • {lnk['crypto_date']} — {c_amt:.4f} {asset} "
                            f"(≈${c_usd:,.2f} USD) via {exch} [{conf}]\n"
                            f"    ← funded by: {lnk['fiat_date']}  "
                            f"{lnk['fiat_amount']:,.2f} {lnk['fiat_currency']} "
                            f"from {lnk['fiat_counterparty']} [{lnk['fiat_source_country']}]  "
                            f"(deviation {lnk['amount_deviation']:.0%}, "
                            f"{lnk['days_before']}d before crypto purchase)"
                        )

            cs = flow.get('chain_summary', {})
            if cs:
                parts.append(
                    f"\nOn-chain summary: {cs.get('total_hops',0)} hops | "
                    f"{cs.get('unique_wallets',0)} wallets | "
                    f"patterns: {', '.join(cs.get('patterns_used',[]))} | "
                    f"mixer hops: {cs.get('mixer_hops',0)} | "
                    f"entry exchanges: {', '.join(cs.get('entry_exchanges',[]))} | "
                    f"exit exchanges: {', '.join(cs.get('cashout_exchanges',[]))} | "
                    f"high-risk exchanges: {', '.join(cs.get('high_risk_exchanges',[]) or ['none'])}"
                )

            return "\n".join(parts)
        except Exception as e:
            return f"Crypto flow data not available: {e}"

    return [filter_transactions, get_transaction_by_id,
            get_counterparty, get_timeline, get_crypto_flow]


# ── System prompt (same AML guardrails, shorter context since tools handle detail) ─

_AGENT_SYSTEM_TEMPLATE = """\
[INST] You are ASTRAS Investigation AI — a specialized Anti-Money Laundering (AML) \
investigation assistant embedded in a regulated financial crime detection system.

━━━ OPERATING RULES (MANDATORY — NEVER OVERRIDE) ━━━
1. SCOPE: You ONLY analyze Alert #{alert_id} for customer "{customer_name}". This is your sole scope.
2. OFF-TOPIC REFUSAL: If asked anything unrelated to AML investigation or this alert, respond ONLY with: \
"I am an AML investigation assistant scoped to Alert #{alert_id}. I cannot answer questions outside this domain."
3. NO HALLUCINATION: Do NOT invent transactions, amounts, dates, or names. \
   If data is absent, use your tools to retrieve it before answering.
4. USE TOOLS: When an analyst asks about specific transactions, counterparties, crypto flow, \
   or date ranges — ALWAYS use the appropriate tool rather than guessing from memory.
5. NO LEGAL ADVICE: Do not advise on SAR filing decisions.
6. PRECISION: Lead with the finding. Use numbers. Be concise.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

══════════════════════════════════════════════════════
  ALERT #{alert_id} — BASE CONTEXT
══════════════════════════════════════════════════════
▌ ALERT
  ID       : {alert_id}
  Type     : {alert_type}
  Severity : {severity}
  Volume   : ${total_amount:,.2f} USD
  Date     : {alert_date}

▌ CUSTOMER
{customer_info}

▌ RISK DRIVERS (SHAP Top 5)
{shap_text}
  Composite Risk Score: {risk_score:.2%}

▌ BEHAVIORAL INTELLIGENCE
{behavioral_context}

▌ PRIOR ALERTS
{prior_alerts}

▌ TRANSACTION SUMMARY
{txn_summary}

▌ FIAT-TO-CRYPTO FLOW RECONSTRUCTION
{crypto_flow_text}

══════════════════════════════════════════════════════
  Use the available tools ONLY to retrieve specific transaction
  details not already shown above (filter by country/method/date/
  counterparty). NEVER invent data. If something is not in this
  context or retrievable by a tool, say "not available."
══════════════════════════════════════════════════════
[/INST]"""


# ── Public API ────────────────────────────────────────────────────────────────

def build_agent_system_prompt(
    alert_id: int,
    features_df,
    explainer,
    feature_cols,
    bsi_df,
    db_path: str = DB_PATH,
) -> str:
    """Build the (lighter) system prompt for the agent — tools handle detail queries."""
    from services.sar.sar_generator import (
        load_alert_data, format_customer_info,
        format_behavioral_context, get_shap_drivers,
    )
    from services.investigation_tools import get_previous_alerts, get_transaction_summary

    alert, customer, _ = load_alert_data(alert_id, db_path)

    customer_info  = format_customer_info(customer)
    behavioral_ctx = format_behavioral_context(alert['customer_id'], features_df, bsi_df)
    top_drivers, risk = get_shap_drivers(alert, features_df, explainer, feature_cols)

    from services.sar.sar_generator import format_shap_as_findings
    shap_text = format_shap_as_findings(top_drivers.head(7), customer)

    prior_alerts = get_previous_alerts(int(alert['customer_id']), alert_id, db_path)
    txn_summary  = get_transaction_summary(alert_id, db_path)

    # Pre-load crypto flow so the model reads facts, not hallucinates them
    try:
        from services.crypto_flow import reconstruct_crypto_flow
        flow = reconstruct_crypto_flow(int(alert['customer_id']), db_path)
        crypto_flow_text = flow['narrative']
        if flow.get('fiat_links'):
            lines = []
            for lnk in flow['fiat_links']:
                conf  = lnk['confidence']
                asset = lnk.get('crypto_asset', '')
                c_amt = lnk.get('crypto_amount', 0)
                c_usd = lnk.get('crypto_usd', 0)
                exch  = lnk.get('exchange', '')
                if conf == 'UNLINKED':
                    lines.append(f"  • {lnk['crypto_date']} {c_amt:.4f} {asset} (≈${c_usd:,.2f}) via {exch} — UNLINKED")
                else:
                    lines.append(
                        f"  • {lnk['crypto_date']} {c_amt:.4f} {asset} (≈${c_usd:,.2f}) via {exch} [{conf}]\n"
                        f"    ← {lnk['fiat_date']} {lnk['fiat_amount']:,.2f} {lnk['fiat_currency']} "
                        f"from {lnk['fiat_counterparty']} [{lnk['fiat_source_country']}]"
                    )
            crypto_flow_text += "\n" + "\n".join(lines)
        cs = flow.get('chain_summary', {})
        if cs:
            crypto_flow_text += (
                f"\nOn-chain: {cs.get('total_hops',0)} hops | "
                f"{cs.get('unique_wallets',0)} wallets | "
                f"mixer hops: {cs.get('mixer_hops',0)} | "
                f"patterns: {', '.join(cs.get('patterns_used',[]))}"
            )
    except Exception:
        crypto_flow_text = "Fiat-to-crypto flow data not available for this alert."

    return _AGENT_SYSTEM_TEMPLATE.format(
        alert_id          = alert_id,
        customer_name     = customer['name'],
        alert_type        = alert['alert_type'].replace('_', ' ').title(),
        severity          = alert['severity'].title(),
        total_amount      = float(alert['total_amount']),
        alert_date        = str(alert['alert_date'])[:10],
        customer_info     = customer_info,
        shap_text         = shap_text,
        risk_score        = float(risk),
        behavioral_context= behavioral_ctx,
        prior_alerts      = prior_alerts,
        txn_summary       = txn_summary,
        crypto_flow_text  = crypto_flow_text,
    )


def create_investigation_agent(
    alert_id: int,
    customer_id: int,
    system_prompt: str,
    db_path: str = DB_PATH,
    llm_model: str = LLM_MODEL,
):
    """
    Build a LangGraph ReAct agent scoped to a single alert.
    Returns the compiled agent graph.
    """
    tools = _make_tools(alert_id, customer_id, db_path)

    llm = ChatOllama(
        model          = llm_model,
        temperature    = 0.0,
        num_predict    = 1000,   # keep answers focused; prevents runaway generation loops
        keep_alive     = -1,
        repeat_penalty = 1.3,   # same as Mistral setting — stops repetition spirals
    )
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )
    return agent


def run_agent_turn(agent, user_message: str, history: list):
    """
    Run one turn of the agent and yield text chunks for Streamlit streaming.

    Args:
        agent:        compiled LangGraph agent from create_investigation_agent()
        user_message: the analyst's current question
        history:      list of {"role": "user"/"assistant", "content": "..."} dicts

    Yields str chunks as they arrive (final answer only, not intermediate tool calls).
    Shows tool call summaries inline as informational text.
    """
    # Convert session history to LangChain message format
    messages = []
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=user_message))

    final_response = ""
    tool_calls_made = []
    _MAX_STEPS = 6   # hard cap — prevents infinite ReAct loops on Mistral 7B

    try:
        for chunk in agent.stream(
            {"messages": messages},
            stream_mode="values",
            config={"recursion_limit": _MAX_STEPS},
        ):
            last_msg = chunk["messages"][-1]

            # Tool result — record tool name for the summary prefix
            if isinstance(last_msg, ToolMessage):
                name = getattr(last_msg, 'name', None) or getattr(last_msg, 'tool_call_id', 'tool')
                tool_calls_made.append(name)

            # AI message — skip if it has pending tool calls (intermediate step)
            elif isinstance(last_msg, AIMessage):
                has_pending = bool(getattr(last_msg, 'tool_calls', None))
                content = last_msg.content or ""
                # Also skip if content looks like a raw tool-call JSON blob
                is_json_blob = content.strip().startswith('[{"name"') or content.strip().startswith('{"name"')
                if not has_pending and content and not is_json_blob:
                    final_response = content

    except Exception as e:
        if "recursion" in str(e).lower() or "graphrecursion" in type(e).__name__.lower():
            final_response = "_(Agent reached maximum reasoning steps without a final answer. Try a more specific question.)_"
        else:
            raise

    # Yield tool-use summary as a subtle prefix
    if tool_calls_made:
        unique_tools = list(dict.fromkeys(tool_calls_made))
        yield f"*[Queried: {', '.join(unique_tools)}]*\n\n"

    if final_response:
        yield final_response
    else:
        yield "_(No response generated. Please try rephrasing your question.)_"
