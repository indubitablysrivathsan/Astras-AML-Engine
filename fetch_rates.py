"""
Run this separately to download live FX + crypto rates from Yahoo Finance.
Saves to data/rates_cache.csv which the pipeline will use on next run.

Usage:
    python fetch_rates.py
"""
import os
import sys
import pandas as pd
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE_STR, SIMULATION_DAYS, DATA_DIR, FX_RATES, CRYPTO_RATES_USD
from datetime import datetime, timedelta

TICKERS = {
    'EUR':  'EURUSD=X',
    'AED':  'AEDUSD=X',
    'INR':  'INRUSD=X',
    'BTC':  'BTC-USD',
    'ETH':  'ETH-USD',
    'USDT': 'USDT-USD',
}
STATIC_FALLBACK = {**FX_RATES, **{k: v for k, v in CRYPTO_RATES_USD.items()}, 'USD': 1.0}
CACHE_PATH = os.path.join(DATA_DIR, 'rates_cache.csv')

start = START_DATE_STR
end = (datetime.strptime(start, '%Y-%m-%d') + timedelta(days=SIMULATION_DAYS + 5)).strftime('%Y-%m-%d')

print(f"Fetching rates {start} → {end} ...")

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

def _fetch(currency, ticker):
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError("empty response")
    close = raw['Close'].copy()
    close.index = pd.to_datetime(close.index).normalize()
    close.name = currency
    return close

frames = []
for currency, ticker in TICKERS.items():
    idx = pd.date_range(start=start, end=end, freq='D')
    fallback = pd.Series(STATIC_FALLBACK.get(currency, 1.0), index=idx, name=currency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch, currency, ticker)
        try:
            frames.append(future.result(timeout=30))
            print(f"  {currency} OK")
        except concurrent.futures.TimeoutError:
            print(f"  {currency} timed out — using static fallback")
            frames.append(fallback)
        except Exception as e:
            print(f"  {currency} failed ({e}) — using static fallback")
            frames.append(fallback)

combined = pd.concat(frames, axis=1)
combined.index.name = 'date'
combined = combined.ffill().bfill()
combined['USD'] = 1.0
combined.reset_index(inplace=True)

os.makedirs(DATA_DIR, exist_ok=True)
combined.to_csv(CACHE_PATH, index=False)
print(f"\nSaved to {CACHE_PATH}")
