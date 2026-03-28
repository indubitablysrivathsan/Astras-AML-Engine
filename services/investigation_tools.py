"""
Investigation Tools — Alert-scoped data access for the investigation chatbot.
All functions are read-only and scoped to a single alert / customer.
Used as Tier-2 context (on-demand) and for the CSV download button.
"""
import sqlite3
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH


# ── Core logical view ─────────────────────────────────────────────────────────

def get_alert_transactions(
    alert_id: int,
    db_path: str = DB_PATH,
    country: str = None,
    method: str = None,
    txn_type: str = None,
    min_amount: float = None,
    max_amount: float = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Return transactions for the alert's customer.
    Acts as a logical view — parameterized, SELECT-only, no physical SQLite view created.
    Optional filters: country, method, txn_type, min/max amount.
    """
    conn = sqlite3.connect(db_path)
    alert_row = pd.read_sql(
        "SELECT customer_id FROM alerts WHERE alert_id = ?",
        conn, params=(alert_id,)
    )
    if alert_row.empty:
        conn.close()
        return pd.DataFrame()

    customer_id = int(alert_row.iloc[0]['customer_id'])

    query = """
        SELECT transaction_id, transaction_date, transaction_type,
               amount, currency, usd_rate, method, location,
               country, counterparty, counterparty_bank, crypto_asset, description
        FROM transactions
        WHERE customer_id = ?
    """
    params = [customer_id]

    if country:
        query += " AND country = ?"
        params.append(country)
    if method:
        query += " AND method = ?"
        params.append(method)
    if txn_type:
        query += " AND transaction_type = ?"
        params.append(txn_type)
    if min_amount is not None:
        query += " AND amount >= ?"
        params.append(min_amount)
    if max_amount is not None:
        query += " AND amount <= ?"
        params.append(max_amount)

    query += " ORDER BY transaction_date DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


def get_transaction_summary(alert_id: int, db_path: str = DB_PATH) -> str:
    """Aggregated summary of all transactions for this alert's customer."""
    df = get_alert_transactions(alert_id, db_path, limit=10000)
    if df.empty:
        return "No transactions found."

    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['usd_amount'] = df['amount'] * df.get('usd_rate', 1.0).fillna(1.0) \
        if 'usd_rate' in df.columns else df['amount']

    deps = df[df['transaction_type'] == 'deposit']
    wds  = df[df['transaction_type'] == 'withdrawal']

    lines = [
        f"Total transactions: {len(df)}",
        f"Date range: {df['transaction_date'].min().date()} → {df['transaction_date'].max().date()}",
        f"Deposits:    {len(deps):>4}  totaling ${deps['usd_amount'].sum():>12,.2f} USD",
        f"Withdrawals: {len(wds):>4}  totaling ${wds['usd_amount'].sum():>12,.2f} USD",
        f"Net flow: ${(deps['usd_amount'].sum() - wds['usd_amount'].sum()):,.2f} USD",
        "",
        "By country:",
    ]
    for country, grp in df.groupby('country'):
        lines.append(f"  {country:<30} {len(grp):>4} txns   ${grp['usd_amount'].sum():>12,.2f} USD")

    lines.append("\nBy method:")
    for method, grp in df.groupby('method'):
        lines.append(f"  {method:<30} {len(grp):>4} txns   ${grp['usd_amount'].sum():>12,.2f} USD")

    lines.append("\nBy month:")
    for period, grp in df.groupby(df['transaction_date'].dt.to_period('M')):
        lines.append(f"  {str(period):<12} {len(grp):>4} txns   ${grp['usd_amount'].sum():>12,.2f} USD")

    if 'currency' in df.columns:
        currencies = df['currency'].value_counts()
        lines.append("\nCurrencies used: " + ", ".join(
            f"{cur} ({cnt})" for cur, cnt in currencies.items()
        ))

    return "\n".join(lines)


def get_previous_alerts(customer_id: int, current_alert_id: int, db_path: str = DB_PATH) -> str:
    """Return prior alert history for the customer (excluding the current alert)."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """SELECT alert_id, alert_type, severity, total_amount, alert_date
           FROM alerts
           WHERE customer_id = ? AND alert_id != ?
           ORDER BY alert_date DESC""",
        conn, params=(customer_id, current_alert_id)
    )
    conn.close()

    if df.empty:
        return "No prior alerts on record for this customer."

    lines = [f"Prior alerts for this customer ({len(df)} total):"]
    for _, row in df.iterrows():
        lines.append(
            f"  Alert {row['alert_id']}: {row['alert_type'].replace('_',' ').title()} | "
            f"{row['severity'].title()} | ${row['total_amount']:,.2f} | {row['alert_date'][:10]}"
        )
    return "\n".join(lines)


def format_transactions_for_context(df: pd.DataFrame, limit: int = 100) -> str:
    """Format a transactions DataFrame as readable text for LLM context."""
    if df.empty:
        return "No transactions."

    if 'usd_rate' in df.columns:
        df = df.copy()
        df['usd_amount'] = df['amount'] * df['usd_rate'].fillna(1.0)

    lines = []
    for _, txn in df.head(limit).iterrows():
        cur = txn.get('currency', 'USD')
        amt = txn['amount']
        if cur != 'USD' and 'usd_amount' in df.columns:
            amt_str = f"{amt:,.8f} {cur} (≈${txn['usd_amount']:,.2f} USD)"
        else:
            amt_str = f"${amt:,.2f}"

        line = (f"- {str(txn['transaction_date'])[:10]}  "
                f"{txn['transaction_type'].title():<12} {amt_str:<35} "
                f"via {txn['method']}")

        if pd.notna(txn.get('counterparty')) and str(txn['counterparty']) not in ('', 'None', 'nan'):
            line += f" | {txn['counterparty']}"
        if txn.get('country') and txn['country'] != 'USA':
            line += f" [{txn['country']}]"
        if pd.notna(txn.get('crypto_asset')) and str(txn.get('crypto_asset')) not in ('', 'None', 'nan'):
            line += f" ({txn['crypto_asset']})"
        lines.append(line)

    if len(df) > limit:
        lines.append(f"\n... {len(df) - limit} more transactions not shown. "
                     "Ask for filtered view (by country, method, type, or amount range).")
    return "\n".join(lines)
