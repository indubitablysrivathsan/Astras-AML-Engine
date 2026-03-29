"""
Phase 1 Only Runner
Regenerates the SQLite database (customers, transactions, alerts,
crypto_chain_txns, crypto_wallets, crypto_fiat_links) without touching
any downstream outputs (CSVs, model, SHAP explainer, ChromaDB).

Run this from WSL when you only need to refresh the DB:
    cd /mnt/c/Users/abhij/Barclay/PS5_Final
    python run_phase1_only.py

The DB is written directly to the Windows path via /mnt/c/ — no file
copying needed. All other pipeline outputs remain valid because
data_generator.py uses fixed seeds (numpy 42, random 42).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_phase1():
    start = time.time()
    print("=" * 60)
    print("  ASTRAS — Phase 1: Data Generation (DB refresh only)")
    print("=" * 60)

    from services.data_generation.data_generator import run as generate_data
    customers_df, transactions_df, alerts_df = generate_data()

    elapsed = time.time() - start
    print(f"\nPhase 1 complete in {elapsed:.1f}s")
    print(f"  Customers:    {len(customers_df):,}")
    print(f"  Transactions: {len(transactions_df):,}")
    print(f"  Alerts:       {len(alerts_df):,}")

    # Confirm crypto_fiat_links was written
    import sqlite3
    from config import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    try:
        count = conn.execute("SELECT COUNT(*) FROM crypto_fiat_links").fetchone()[0]
        print(f"  Fiat links:   {count:,}  ✓  crypto_fiat_links table populated")
    except Exception as e:
        print(f"  [WARN] crypto_fiat_links check failed: {e}")
    finally:
        conn.close()

    print("\nDone. No other pipeline phases need to re-run.")
    print("Switch back to Windows and launch the app normally.")


if __name__ == "__main__":
    run_phase1()
