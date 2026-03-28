"""
Historical FX and Crypto Rate Fetcher
Fetches daily close rates from Yahoo Finance for the simulation period,
caches them to a CSV so subsequent runs don't hit the network.

Lookup: get_usd_rate(currency, date) -> float
  Returns the USD equivalent of 1 unit of `currency` on `date`.
  Falls back to the nearest prior trading day, then to static fallback.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import START_DATE_STR, SIMULATION_DAYS, DATA_DIR, FX_RATES, CRYPTO_RATES_USD

# Yahoo Finance tickers for each currency code we support
TICKERS = {
    'EUR':  'EURUSD=X',
    'AED':  'AEDUSD=X',
    'INR':  'INRUSD=X',
    'BTC':  'BTC-USD',
    'ETH':  'ETH-USD',
    'USDT': 'USDT-USD',
}

# Static fallback (used if yfinance fails or currency is unmapped)
STATIC_FALLBACK = {**FX_RATES, **{k: v for k, v in CRYPTO_RATES_USD.items()}}
STATIC_FALLBACK.setdefault('USD', 1.0)

CACHE_PATH = os.path.join(DATA_DIR, 'rates_cache.csv')

# Module-level lookup table: {date_str -> {currency -> rate}}
_rate_table: dict[str, dict[str, float]] = {}
_sorted_dates: list[str] = []


def _build_rate_table(start: str, end: str) -> pd.DataFrame:
    """Fetch or load cached daily rates for all supported currencies."""
    # Skip live fetch — use static fallback rates. Run fetch_rates.py separately for live data.
    if not os.path.exists(CACHE_PATH):
        print("  [rates] No cache found — using static fallback rates (run fetch_rates.py for live data)")
        return _static_rate_df(start, end)

    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=['date'])
        # Check the cache covers our simulation window
        if df['date'].min().date().isoformat() <= start and \
           df['date'].max().date().isoformat() >= end:
            print(f"  [rates] Loaded cached rates from {CACHE_PATH}")
            return df

    try:
        import yfinance as yf
    except ImportError:
        print("  [rates] yfinance not installed — using static fallback rates")
        return _static_rate_df(start, end)

    print(f"  [rates] Fetching historical rates from Yahoo Finance ({start} → {end}) …")

    def _fetch(currency, ticker):
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError(f"Empty data for {ticker}")
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
                close = future.result(timeout=15)
                frames.append(close)
                print(f"  [rates]   {currency} OK")
            except concurrent.futures.TimeoutError:
                print(f"  [rates] Warning: {ticker} timed out after 15s, using static fallback")
                frames.append(fallback)
            except Exception as e:
                print(f"  [rates] Warning: could not fetch {ticker} ({e}), using static fallback")
                frames.append(fallback)

    combined = pd.concat(frames, axis=1)
    combined.index.name = 'date'
    # Forward-fill weekends/holidays from last known trading close
    combined = combined.ffill().bfill()
    combined['USD'] = 1.0
    combined.reset_index(inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(CACHE_PATH, index=False)
    print(f"  [rates] Saved to {CACHE_PATH}")
    return combined


def _static_rate_df(start: str, end: str) -> pd.DataFrame:
    """Build a flat rate DataFrame from static config values."""
    idx = pd.date_range(start=start, end=end, freq='D')
    data = {cur: STATIC_FALLBACK.get(cur, 1.0) for cur in list(TICKERS.keys()) + ['USD']}
    df = pd.DataFrame(data, index=idx)
    df.index.name = 'date'
    return df.reset_index()


def load_rates():
    """Load rates into the module-level lookup table. Call once at pipeline start."""
    global _rate_table, _sorted_dates
    if _rate_table:
        return  # already loaded

    start = START_DATE_STR
    from datetime import datetime
    end_dt = datetime.strptime(start, '%Y-%m-%d') + timedelta(days=SIMULATION_DAYS + 5)
    end = end_dt.strftime('%Y-%m-%d')

    df = _build_rate_table(start, end)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    for _, row in df.iterrows():
        d = row['date'].strftime('%Y-%m-%d')
        _rate_table[d] = {
            col: float(row[col])
            for col in df.columns if col != 'date'
        }

    _sorted_dates = sorted(_rate_table.keys())


def get_usd_rate(currency: str, date) -> float:
    """
    Return the USD rate for `currency` on `date` (datetime or str).
    - USD always returns 1.0.
    - Falls back to nearest prior trading day if exact date missing.
    - Falls back to static config if currency unknown.
    """
    if not _rate_table:
        load_rates()

    if currency == 'USD':
        return 1.0

    if hasattr(date, 'strftime'):
        date_str = date.strftime('%Y-%m-%d')
    else:
        date_str = str(date)[:10]

    # Exact match
    if date_str in _rate_table and currency in _rate_table[date_str]:
        return _rate_table[date_str][currency]

    # Nearest prior trading day
    for d in reversed(_sorted_dates):
        if d <= date_str and currency in _rate_table[d]:
            return _rate_table[d][currency]

    # Static fallback
    return STATIC_FALLBACK.get(currency, 1.0)


def to_usd(amount: float, currency: str, date) -> float:
    """Convert `amount` in `currency` to USD on `date`."""
    return round(amount * get_usd_rate(currency, date), 6)
