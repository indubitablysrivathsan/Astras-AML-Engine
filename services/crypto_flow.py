"""
Crypto Flow Reconstruction
Traces fiat → crypto → fiat laundering paths for crypto_laundering alerts.

Linkage methodology (probabilistic, not deterministic):
  1. Temporal proximity  — bank deposit within N days BEFORE a crypto purchase
  2. Amount similarity   — USD-equivalent of crypto ≈ fiat deposit amount
  3. Confidence tier:
       HIGH   — both temporal AND amount match within tight thresholds
       MEDIUM — temporal match, amount loosely correlated
       LOW    — temporal match only (amount differs significantly)

Outputs a structured dict consumed by sar_fallback.py → SOURCE OF FIAT FUNDS.
"""
import sqlite3
import os
import sys
from datetime import timedelta

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from config import DB_PATH, CRYPTO_RATES_USD, EXCHANGE_PROFILES  # noqa: F401


# ── Configuration ─────────────────────────────────────────────────────────────
_LOOKBACK_DAYS     = 10    # fiat deposit must occur within this window before crypto purchase
_TIGHT_AMT_RATIO   = 0.25  # ≤25% deviation → HIGH confidence
_LOOSE_AMT_RATIO   = 0.60  # ≤60% deviation → MEDIUM confidence


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_customer_bank_txns(customer_id: int, db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """SELECT transaction_id, transaction_date, transaction_type,
                  amount, currency, usd_rate, method, location, country,
                  counterparty, counterparty_bank, crypto_asset
           FROM transactions
           WHERE customer_id = ?
           ORDER BY transaction_date""",
        conn, params=(customer_id,)
    )
    conn.close()
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['usd_amount'] = df['amount'] * df['usd_rate'].fillna(1.0)
    return df


def _load_chain_data(customer_id: int, db_path: str):
    """Load on-chain data.  Returns (chain_df, wallets_df) or (empty, empty)."""
    conn = sqlite3.connect(db_path)
    try:
        chain   = pd.read_sql(
            "SELECT * FROM crypto_chain_txns WHERE customer_id = ?",
            conn, params=(customer_id,)
        )
        wallets = pd.read_sql(
            "SELECT * FROM crypto_wallets WHERE customer_id = ?",
            conn, params=(customer_id,)
        )
    except Exception:
        chain   = pd.DataFrame()
        wallets = pd.DataFrame()
    conn.close()
    return chain, wallets


# ── Core linkage logic ────────────────────────────────────────────────────────

def _load_exact_fiat_links(customer_id: int, db_path: str) -> list:
    """
    Load ground-truth fiat->crypto links from crypto_fiat_links table.
    Returns list of link dicts in the same format as _link_fiat_to_crypto,
    or empty list if the table doesn't exist or has no rows for this customer.
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT * FROM crypto_fiat_links WHERE customer_id = ? ORDER BY cycle_id",
            conn, params=(customer_id,)
        )
        conn.close()
    except Exception:
        return []

    if df.empty:
        return []

    links = []
    for _, row in df.iterrows():
        links.append({
            'crypto_txn_id':       None,
            'crypto_date':         str(row.get('buy_date', '')),
            'crypto_asset':        str(row.get('crypto_asset', '')),
            'crypto_amount':       float(row.get('crypto_amount', 0)),
            'crypto_usd':          float(row.get('fiat_usd', 0)),  # approx
            'exchange':            str(row.get('exchange', 'Unknown')),
            'fiat_txn_id':         None,
            'fiat_date':           str(row.get('fiat_date', '')),
            'fiat_amount':         float(row.get('fiat_amount', 0)),
            'fiat_currency':       str(row.get('fiat_currency', '')),
            'fiat_usd':            float(row.get('fiat_usd', 0)),
            'fiat_source_country': str(row.get('fiat_source_country', '')),
            'fiat_counterparty':   str(row.get('fiat_counterparty', '')),
            'confidence':          'EXACT',
            'days_before':         None,
            'amount_deviation':    0.0,
        })
    return links


def _link_fiat_to_crypto(bank_df: pd.DataFrame) -> list:
    """
    For each crypto purchase (withdrawal via crypto_exchange) find the best
    matching fiat deposit in the preceding _LOOKBACK_DAYS window.
    Returns list of link dicts.
    """
    purchases = bank_df[
        (bank_df['method'] == 'crypto_exchange') &
        (bank_df['transaction_type'] == 'withdrawal')
    ].copy()

    fiat_deps = bank_df[
        (bank_df['transaction_type'] == 'deposit') &
        (bank_df['method'] != 'crypto_exchange')
    ].copy()

    links = []
    for _, pur in purchases.iterrows():
        pur_date     = pur['transaction_date']
        pur_usd      = pur['usd_amount']
        window_start = pur_date - timedelta(days=_LOOKBACK_DAYS)

        candidates = fiat_deps[
            (fiat_deps['transaction_date'] >= window_start) &
            (fiat_deps['transaction_date'] <  pur_date)
        ].copy()

        if candidates.empty:
            links.append({
                'crypto_txn_id':    int(pur['transaction_id']),
                'crypto_date':      str(pur['transaction_date'])[:10],
                'crypto_asset':     str(pur.get('crypto_asset', '')),
                'crypto_amount':    float(pur['amount']),
                'crypto_usd':       round(float(pur_usd), 2),
                'exchange':         str(pur.get('location', 'Unknown')),
                'fiat_txn_id':      None,
                'fiat_date':        None,
                'fiat_amount':      None,
                'fiat_currency':    None,
                'fiat_usd':         None,
                'fiat_source_country': None,
                'fiat_counterparty':   None,
                'confidence':       'UNLINKED',
                'days_before':      None,
                'amount_deviation': None,
            })
            continue

        # Pick closest in time (last deposit before purchase)
        candidates = candidates.sort_values('transaction_date')
        best = candidates.iloc[-1]

        days_before = (pur_date - best['transaction_date']).days
        fiat_usd    = float(best['usd_amount'])
        deviation   = abs(fiat_usd - float(pur_usd)) / max(float(pur_usd), 1)

        if deviation <= _TIGHT_AMT_RATIO:
            confidence = 'HIGH'
        elif deviation <= _LOOSE_AMT_RATIO:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        links.append({
            'crypto_txn_id':       int(pur['transaction_id']),
            'crypto_date':         str(pur['transaction_date'])[:10],
            'crypto_asset':        str(pur.get('crypto_asset', '')),
            'crypto_amount':       float(pur['amount']),
            'crypto_usd':          round(float(pur_usd), 2),
            'exchange':            str(pur.get('location', 'Unknown')),
            'fiat_txn_id':         int(best['transaction_id']),
            'fiat_date':           str(best['transaction_date'])[:10],
            'fiat_amount':         round(float(best['amount']), 2),
            'fiat_currency':       str(best['currency']),
            'fiat_usd':            round(fiat_usd, 2),
            'fiat_source_country': str(best['country']),
            'fiat_counterparty':   str(best.get('counterparty', '')),
            'confidence':          confidence,
            'days_before':         days_before,
            'amount_deviation':    round(deviation, 3),
        })

    return links


# ── On-chain hop summary ──────────────────────────────────────────────────────

def _summarise_chain(chain_df: pd.DataFrame, wallets_df: pd.DataFrame) -> dict:
    """Extract key on-chain metrics for the SAR narrative."""
    if chain_df.empty:
        return {}

    total_hops      = len(chain_df)
    patterns_used   = chain_df['pattern_type'].unique().tolist()
    mixer_hops      = int(chain_df['is_mixer'].sum())
    cashout_hops    = int(chain_df['is_cashout'].sum())
    unique_wallets  = len(set(chain_df['from_address'].tolist() + chain_df['to_address'].tolist()))
    total_fees      = round(float(chain_df['fee_amount'].sum()), 8)

    exchange_wallets = wallets_df[wallets_df['entity_type'].str.contains('exchange', na=False)] \
        if not wallets_df.empty else pd.DataFrame()

    entry_exchanges  = wallets_df[wallets_df['is_entry_point']  == True]['exchange_name'].dropna().unique().tolist() \
        if not wallets_df.empty else []
    cashout_exchanges= wallets_df[wallets_df['is_cashout_point'] == True]['exchange_name'].dropna().unique().tolist() \
        if not wallets_df.empty else []

    high_risk = [e for e in (entry_exchanges + cashout_exchanges)
                 if EXCHANGE_PROFILES.get(e, {}).get('risk_score', 0) >= 0.50]

    return {
        'total_hops':         total_hops,
        'unique_wallets':     unique_wallets,
        'patterns_used':      patterns_used,
        'mixer_hops':         mixer_hops,
        'cashout_hops':       cashout_hops,
        'total_fees_crypto':  total_fees,
        'entry_exchanges':    entry_exchanges,
        'cashout_exchanges':  cashout_exchanges,
        'high_risk_exchanges': list(set(high_risk)),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def reconstruct_crypto_flow(customer_id: int, db_path: str = DB_PATH) -> dict:
    """
    Full fiat→crypto flow reconstruction for a single customer.

    Returns a structured dict with:
      - fiat_links:   list of link objects (one per crypto purchase)
      - chain_summary: on-chain hop stats
      - narrative:    pre-built SOURCE OF FIAT FUNDS paragraph (for SAR)
      - has_chain_data: bool (True if crypto_chain_txns table was populated)
    """
    bank_df              = _load_customer_bank_txns(customer_id, db_path)
    chain_df, wallets_df = _load_chain_data(customer_id, db_path)

    # Use ground-truth links if available; fall back to probabilistic inference
    exact_links = _load_exact_fiat_links(customer_id, db_path)
    links       = exact_links if exact_links else _link_fiat_to_crypto(bank_df)
    chain_summary = _summarise_chain(chain_df, wallets_df)

    narrative = _build_fiat_funds_narrative(customer_id, bank_df, links, chain_summary)

    return {
        'customer_id':   customer_id,
        'fiat_links':    links,
        'chain_summary': chain_summary,
        'has_chain_data': not chain_df.empty,
        'narrative':     narrative,
    }


def _build_fiat_funds_narrative(customer_id, bank_df, links, chain_summary) -> str:
    """
    Build the SOURCE OF FIAT FUNDS paragraph per FinCEN virtual-currency SAR guidance.
    Uses structured chain data where available; gracefully degrades to aggregated stats.
    """
    crypto_purchases = bank_df[
        (bank_df['method'] == 'crypto_exchange') &
        (bank_df['transaction_type'] == 'withdrawal')
    ]
    if crypto_purchases.empty:
        return "No cryptocurrency exchange transactions identified for this customer."

    total_crypto_usd = float(crypto_purchases['usd_amount'].sum())
    assets_used = crypto_purchases['crypto_asset'].dropna().unique().tolist()
    assets_str  = ', '.join(assets_used) if assets_used else 'Unknown'

    # Exchange names from bank txns
    exchanges = crypto_purchases['location'].dropna().unique().tolist()
    exch_str  = ', '.join(exchanges[:5]) if exchanges else 'Unknown'

    # Linked fiat deposits
    linked = [l for l in links if l['confidence'] in ('HIGH', 'MEDIUM', 'LOW')]
    unlinked_count = sum(1 for l in links if l['confidence'] == 'UNLINKED')

    if linked:
        linked_usd = sum(l['fiat_usd'] for l in linked if l['fiat_usd'])
        high_conf  = [l for l in linked if l['confidence'] == 'HIGH']
        med_conf   = [l for l in linked if l['confidence'] == 'MEDIUM']
        source_countries = list({l['fiat_source_country'] for l in linked
                                  if l['fiat_source_country'] and l['fiat_source_country'] != 'None'})
        currencies  = list({l['fiat_currency'] for l in linked
                            if l['fiat_currency'] and l['fiat_currency'] != 'None'})
        counterparties = list({l['fiat_counterparty'] for l in linked
                               if l['fiat_counterparty'] and l['fiat_counterparty'] not in ('None', '', 'nan')})[:4]

        conf_summary = (f"{len(high_conf)} transaction(s) with HIGH confidence linkage and "
                        f"{len(med_conf)} with MEDIUM confidence")

        fiat_origin = (f"Fiat deposits totaling ${linked_usd:,.2f} USD (equivalent) received via "
                       f"wire transfer from counterparties in {', '.join(source_countries[:5]) or 'multiple jurisdictions'}, "
                       f"denominated in {', '.join(currencies) or 'multiple currencies'}.")
        if counterparties:
            fiat_origin += (f" Identified remitting entities include: "
                            f"{'; '.join(counterparties)}.")
        fiat_origin += (f" Linkage established for {len(linked)} of {len(links)} crypto purchase(s) "
                        f"({conf_summary}).")
        if unlinked_count:
            fiat_origin += (f" {unlinked_count} purchase(s) could not be linked to a specific fiat "
                            f"deposit within the {_LOOKBACK_DAYS}-day lookback window.")
    else:
        fiat_origin = (f"Fiat-to-crypto linkage could not be established within the {_LOOKBACK_DAYS}-day "
                       f"lookback window for any of the {len(links)} crypto purchases identified. "
                       f"The source of fiat funds used to acquire ${total_crypto_usd:,.2f} USD equivalent "
                       f"in digital assets could not be independently verified.")

    # On-chain hop narrative
    if chain_summary:
        patterns_str = ', '.join(chain_summary.get('patterns_used', []))
        mixer_note   = (f" The on-chain path traversed {chain_summary['total_hops']} hop(s) across "
                        f"{chain_summary['unique_wallets']} unique wallet addresses using {patterns_str} pattern(s).")
        if chain_summary.get('mixer_hops', 0) > 0:
            mixer_note += (f" {chain_summary['mixer_hops']} mixer/coinjoin node transaction(s) were "
                           f"identified, consistent with deliberate transaction graph obfuscation.")
        if chain_summary.get('high_risk_exchanges'):
            mixer_note += (f" High-risk or low-KYC exchanges involved: "
                           f"{', '.join(chain_summary['high_risk_exchanges'])}.")
        entry_exit_diff = bool(
            set(chain_summary.get('entry_exchanges', [])) !=
            set(chain_summary.get('cashout_exchanges', []))
        )
        if entry_exit_diff:
            mixer_note += (" Funds exited through a different exchange than the entry point, "
                           "indicative of deliberate KYC arbitrage.")
    else:
        mixer_note = (f" The digital assets ({assets_str}) were transacted through cryptocurrency "
                      f"exchanges including {exch_str}.")

    paragraph = (
        f"SOURCE OF FIAT FUNDS\n"
        f"Pursuant to FinCEN regulatory guidance on virtual currency transactions (FIN-2019-G001), "
        f"the institution has performed a fiat-to-crypto flow reconstruction for this account. "
        f"A total of ${total_crypto_usd:,.2f} USD equivalent was converted into digital assets "
        f"({assets_str}) across {len(links)} exchange transaction(s). "
        f"{fiat_origin}"
        f"{mixer_note} "
        f"The source of the original fiat funds could not be independently verified beyond "
        f"the immediate remitting institution. Further investigation and SAR filing determination "
        f"should be escalated to the designated compliance officer."
    )

    return paragraph
