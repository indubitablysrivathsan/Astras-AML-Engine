"""
Crypto Chain Emulator
Generates synthetic on-chain transaction data for crypto_laundering customers.
Emulates real blockchain behaviour: wallets, hops, mixer nodes, multi-path smurfing,
and round-trip exit patterns.

Each customer gets a private wallet registry — zero cross-customer address reuse.
Four laundering patterns mirror what FATF/FinCEN call typologies in practice:
  1. layering    – linear chain of hops through intermediary wallets
  2. mixing      – funds pass through a mixer / coinjoin node then fan out
  3. smurfing    – parallel low-value paths that reconverge at a collector wallet
  4. round_trip  – funds exit to fiat via a different exchange than they entered
"""
import hashlib
import os
import random
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)
from config import (
    CRYPTO_RATES_USD,
    SUPPORTED_CRYPTO,
    START_DATE_STR,
    EXCHANGE_PROFILES,
)

# High-risk exchanges preferenced by launderers (low KYC)
_HIGH_RISK_EXCHANGES  = [k for k, v in EXCHANGE_PROFILES.items() if v['risk_score'] >= 0.50]
_LOW_RISK_EXCHANGES   = [k for k, v in EXCHANGE_PROFILES.items() if v['risk_score'] <  0.25]

# Typical network fee fractions per asset
_FEE_RATE = {'BTC': 0.0005, 'ETH': 0.002, 'USDT': 0.001}

# Block height simulation: ~144 blocks/day for BTC, ~6500/day for ETH
_BLOCKS_PER_DAY = {'BTC': 144, 'ETH': 6500, 'USDT': 6500}

GENESIS_DATE = '2020-01-01'   # reference for simulated block height


# ── Wallet address generators ─────────────────────────────────────────────────

def _btc_bech32(seed: str) -> str:
    """Generate a deterministic-looking BTC bech32 (native SegWit) address."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    # bech32 addresses are 42 chars: bc1q + 38 alphanumeric (lowercase, no b/i/o)
    charset = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l'
    body = ''.join(charset[int(h[i:i+2], 16) % len(charset)] for i in range(0, 38 * 2, 2))
    return 'bc1q' + body[:38]


def _btc_legacy(seed: str) -> str:
    """Generate a deterministic-looking BTC legacy (P2PKH) address."""
    h = hashlib.md5(seed.encode()).hexdigest()
    charset = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    body = ''.join(charset[int(h[i:i+2], 16) % len(charset)] for i in range(0, 33 * 2, 2))
    return '1' + body[:33]


def _eth_address(seed: str) -> str:
    """Generate a deterministic-looking ETH address."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    return '0x' + h[:40]


def _wallet_address(asset: str, seed: str) -> str:
    if asset == 'BTC':
        return _btc_bech32(seed) if random.random() < 0.6 else _btc_legacy(seed)
    return _eth_address(seed)   # ETH and USDT (ERC-20) both use ETH addresses


def _block_height(timestamp, asset: str) -> int:
    """Approximate block height at a given datetime."""
    from datetime import datetime
    genesis = datetime.strptime(GENESIS_DATE, '%Y-%m-%d')
    delta_days = (timestamp - genesis).total_seconds() / 86400
    return int(delta_days * _BLOCKS_PER_DAY.get(asset, 144))


# ── Wallet registry ───────────────────────────────────────────────────────────

class _WalletRegistry:
    """Per-customer wallet book.  Avoids any cross-customer address collision."""

    def __init__(self, customer_id: int, asset: str, rng: random.Random):
        self._cid   = customer_id
        self._asset = asset
        self._rng   = rng
        self._counter = 0
        self._wallets: dict[str, dict] = {}   # address → wallet record

    def _new_address(self, entity_type: str, exchange_name: str = None) -> str:
        seed = f"{self._cid}-{self._asset}-{self._counter}-{entity_type}"
        self._counter += 1
        addr = _wallet_address(self._asset, seed)
        self._wallets[addr] = {
            'wallet_id':        f'w{self._cid}_{self._counter}',
            'customer_id':      self._cid,
            'address':          addr,
            'asset':            self._asset,
            'entity_type':      entity_type,
            'exchange_name':    exchange_name,
            'risk_score':       EXCHANGE_PROFILES.get(exchange_name, {}).get('risk_score', 0.5)
                                if exchange_name else 0.5,
            'is_entry_point':   entity_type == 'exchange_deposit',
            'is_cashout_point': entity_type == 'exchange_cashout',
            'total_received':   0.0,
            'total_sent':       0.0,
            'first_seen':       None,
            'last_seen':        None,
        }
        return addr

    def entry_wallet(self, exchange: str)   -> str: return self._new_address('exchange_deposit',  exchange)
    def cashout_wallet(self, exchange: str) -> str: return self._new_address('exchange_cashout', exchange)
    def hot_wallet(self)                    -> str: return self._new_address('hot_wallet')
    def mixer_wallet(self)                  -> str: return self._new_address('mixer_node')
    def intermediary(self)                  -> str: return self._new_address('intermediary')

    def record_flow(self, addr: str, amount: float, ts):
        w = self._wallets.get(addr)
        if w is None:
            return
        if w['first_seen'] is None or ts < w['first_seen']:
            w['first_seen'] = ts
        if w['last_seen'] is None or ts > w['last_seen']:
            w['last_seen'] = ts

    def to_dataframe(self) -> pd.DataFrame:
        if not self._wallets:
            return pd.DataFrame()
        return pd.DataFrame(list(self._wallets.values()))


# ── Chain transaction builder ─────────────────────────────────────────────────

def _chain_txn(
    chain_txn_id: str,
    customer_id: int,
    from_addr: str,
    to_addr: str,
    asset: str,
    amount: float,
    timestamp,
    hop: int,
    pattern: str,
    bank_txn_id=None,
    is_mixer: bool = False,
    is_cashout: bool = False,
    registry: _WalletRegistry = None,
) -> dict:
    fee = amount * _FEE_RATE.get(asset, 0.001)
    net = amount - fee
    if registry:
        registry.record_flow(from_addr, amount, timestamp)
        registry.record_flow(to_addr,   net,    timestamp)
    return {
        'chain_txn_id':      chain_txn_id,
        'customer_id':       customer_id,
        'from_address':      from_addr,
        'to_address':        to_addr,
        'asset':             asset,
        'amount':            round(amount, 8),
        'net_amount':        round(net, 8),
        'fee_amount':        round(fee, 8),
        'timestamp':         timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'block_height':      _block_height(timestamp, asset),
        'hop_number':        hop,
        'pattern_type':      pattern,
        'linked_bank_txn_id': bank_txn_id,
        'is_mixer':          is_mixer,
        'is_cashout':        is_cashout,
    }


# ── Pattern generators ────────────────────────────────────────────────────────

def _generate_layering(cid, asset, bank_txn_id, entry_time, amount,
                       entry_exchange, exit_exchange, reg, rng, counter) -> list:
    """
    Linear hop chain: exchange_deposit → hot_wallet_1 → … → hot_wallet_N → exchange_cashout
    Each hop peels off fees.  3-6 hops typical.
    """
    txns  = []
    entry = reg.entry_wallet(entry_exchange)
    hops  = [reg.hot_wallet() for _ in range(rng.randint(3, 6))]
    exit_ = reg.cashout_wallet(exit_exchange)

    wallets = [entry] + hops + [exit_]
    cur_amt = amount
    cur_ts  = entry_time

    for h, (src, dst) in enumerate(zip(wallets[:-1], wallets[1:])):
        delay   = timedelta(hours=rng.randint(1, 72))
        cur_ts += delay
        tid = f"chain_{cid}_{counter}_{h}"
        txns.append(_chain_txn(
            tid, cid, src, dst, asset, cur_amt, cur_ts,
            hop=h, pattern='layering',
            bank_txn_id=(bank_txn_id if h == 0 else None),
            is_cashout=(h == len(wallets) - 2),
            registry=reg,
        ))
        cur_amt = cur_amt * (1 - _FEE_RATE.get(asset, 0.001))  # net after fee

    return txns


def _generate_mixing(cid, asset, bank_txn_id, entry_time, amount,
                     entry_exchange, exit_exchange, reg, rng, counter) -> list:
    """
    Funds enter → mixer node → fan out to N outputs → reconverge at collector → cashout.
    Mixer node is flagged is_mixer=True.
    """
    txns     = []
    entry    = reg.entry_wallet(entry_exchange)
    mixer    = reg.mixer_wallet()
    n_out    = rng.randint(3, 6)
    outputs  = [reg.intermediary() for _ in range(n_out)]
    collector = reg.hot_wallet()
    exit_    = reg.cashout_wallet(exit_exchange)

    cur_ts = entry_time

    # entry → mixer
    delay = timedelta(hours=rng.randint(1, 24))
    cur_ts += delay
    txns.append(_chain_txn(f"chain_{cid}_{counter}_0", cid, entry, mixer, asset,
                           amount, cur_ts, hop=0, pattern='mixing',
                           bank_txn_id=bank_txn_id, is_mixer=False, registry=reg))

    # mixer → N output wallets (slightly different amounts to obscure linkage)
    slice_base = amount / n_out
    for i, out_w in enumerate(outputs):
        delay   = timedelta(minutes=rng.randint(5, 120))
        cur_ts += delay
        slice_  = slice_base * rng.uniform(0.8, 1.2)
        txns.append(_chain_txn(f"chain_{cid}_{counter}_m{i}", cid, mixer, out_w, asset,
                               slice_, cur_ts, hop=1, pattern='mixing',
                               is_mixer=True, registry=reg))

    # output wallets → collector
    for i, out_w in enumerate(outputs):
        delay   = timedelta(hours=rng.randint(12, 96))
        cur_ts += delay
        recv = slice_base * (1 - _FEE_RATE.get(asset, 0.001))
        txns.append(_chain_txn(f"chain_{cid}_{counter}_c{i}", cid, out_w, collector, asset,
                               recv, cur_ts, hop=2, pattern='mixing', registry=reg))

    # collector → exit
    final = slice_base * n_out * (1 - _FEE_RATE.get(asset, 0.001)) ** 2
    cur_ts += timedelta(hours=rng.randint(6, 48))
    txns.append(_chain_txn(f"chain_{cid}_{counter}_exit", cid, collector, exit_, asset,
                           final, cur_ts, hop=3, pattern='mixing',
                           is_cashout=True, registry=reg))

    return txns


def _generate_smurfing(cid, asset, bank_txn_id, entry_time, amount,
                       entry_exchange, exit_exchange, reg, rng, counter) -> list:
    """
    Smurfing / structuring on-chain: amount split into N small transfers
    via parallel wallets that all converge at the same collector.
    """
    txns   = []
    entry  = reg.entry_wallet(entry_exchange)
    n_path = rng.randint(4, 8)
    paths  = [reg.intermediary() for _ in range(n_path)]
    collector = reg.hot_wallet()
    exit_  = reg.cashout_wallet(exit_exchange)

    # entry → parallel intermediaries
    slice_ = amount / n_path
    for i, p in enumerate(paths):
        ts_i = entry_time + timedelta(minutes=rng.randint(0, 240))
        txns.append(_chain_txn(f"chain_{cid}_{counter}_s{i}", cid, entry, p, asset,
                               slice_, ts_i, hop=0, pattern='smurfing',
                               bank_txn_id=(bank_txn_id if i == 0 else None), registry=reg))

    # parallel intermediaries → collector (staggered arrivals)
    reconverge_base = entry_time + timedelta(days=rng.randint(1, 7))
    for i, p in enumerate(paths):
        ts_r = reconverge_base + timedelta(hours=rng.randint(0, 48))
        recv = slice_ * (1 - _FEE_RATE.get(asset, 0.001))
        txns.append(_chain_txn(f"chain_{cid}_{counter}_r{i}", cid, p, collector, asset,
                               recv, ts_r, hop=1, pattern='smurfing', registry=reg))

    # collector → cashout
    final = slice_ * n_path * (1 - _FEE_RATE.get(asset, 0.001)) ** 2
    ts_out = reconverge_base + timedelta(days=rng.randint(1, 5))
    txns.append(_chain_txn(f"chain_{cid}_{counter}_out", cid, collector, exit_, asset,
                           final, ts_out, hop=2, pattern='smurfing',
                           is_cashout=True, registry=reg))

    return txns


def _generate_round_trip(cid, asset, bank_txn_id, entry_time, amount,
                         entry_exchange, exit_exchange, reg, rng, counter) -> list:
    """
    Funds enter via one exchange, hop through a few wallets, and exit via a
    *different* exchange — deliberate KYC arbitrage.
    """
    txns   = []
    entry  = reg.entry_wallet(entry_exchange)
    mid1   = reg.hot_wallet()
    mid2   = reg.intermediary()
    exit_  = reg.cashout_wallet(exit_exchange)   # different exchange

    wallets = [entry, mid1, mid2, exit_]
    cur_amt = amount
    cur_ts  = entry_time

    for h, (src, dst) in enumerate(zip(wallets[:-1], wallets[1:])):
        cur_ts += timedelta(hours=rng.randint(6, 120))
        txns.append(_chain_txn(
            f"chain_{cid}_{counter}_rt{h}", cid, src, dst, asset, cur_amt, cur_ts,
            hop=h, pattern='round_trip',
            bank_txn_id=(bank_txn_id if h == 0 else None),
            is_cashout=(h == len(wallets) - 2),
            registry=reg,
        ))
        cur_amt *= (1 - _FEE_RATE.get(asset, 0.001))

    return txns


_PATTERN_FNS = [_generate_layering, _generate_mixing, _generate_smurfing, _generate_round_trip]


# ── Public entry point ────────────────────────────────────────────────────────

def generate_chain_for_customer(customer: dict, bank_txns_df: pd.DataFrame):
    """
    Build synthetic on-chain data for a single crypto_laundering customer.

    Args:
        customer:      row dict from customers_df (must have customer_id, preferred_crypto)
        bank_txns_df:  ALL bank transactions for this customer (with transaction_id assigned)

    Returns:
        chain_df:   DataFrame of on-chain hops  (crypto_chain_txns table)
        wallets_df: DataFrame of wallet records  (crypto_wallets table)
    """
    cid   = int(customer['customer_id'])
    asset = customer.get('preferred_crypto') or 'BTC'
    if asset not in SUPPORTED_CRYPTO:
        asset = 'BTC'

    rng = random.Random(cid * 31337)   # deterministic per-customer seed
    reg = _WalletRegistry(cid, asset, rng)

    # Identify crypto exchange bank transactions (the on-ramps and off-ramps)
    crypto_txns = bank_txns_df[
        (bank_txns_df['customer_id'] == cid) &
        (bank_txns_df['method'] == 'crypto_exchange')
    ].copy()

    if crypto_txns.empty:
        return pd.DataFrame(), pd.DataFrame()

    crypto_txns['transaction_date'] = pd.to_datetime(crypto_txns['transaction_date'])

    # Pair up: each 'withdrawal' (fiat→crypto purchase) triggers a chain
    purchases = crypto_txns[crypto_txns['transaction_type'] == 'withdrawal'].copy()

    all_chain_txns = []
    counter = 0

    for _, row in purchases.iterrows():
        bank_txn_id  = int(row['transaction_id'])
        entry_time   = row['transaction_date']
        crypto_amount = float(row['amount'])   # already in native crypto units

        # Pick entry exchange from bank txn location; exit from a *different* high-risk one
        entry_exchange = str(row.get('location', 'Binance'))
        if entry_exchange not in EXCHANGE_PROFILES:
            entry_exchange = rng.choice(_HIGH_RISK_EXCHANGES)

        # Exit via a different exchange (KYC arbitrage)
        exit_candidates = [e for e in _HIGH_RISK_EXCHANGES if e != entry_exchange]
        if not exit_candidates:
            exit_candidates = _HIGH_RISK_EXCHANGES
        exit_exchange = rng.choice(exit_candidates)

        # Randomly pick a laundering pattern (weighted toward mixing/smurfing)
        fn = rng.choices(
            _PATTERN_FNS,
            weights=[0.25, 0.30, 0.25, 0.20],
            k=1
        )[0]

        hops = fn(cid, asset, bank_txn_id, entry_time, crypto_amount,
                  entry_exchange, exit_exchange, reg, rng, counter)
        all_chain_txns.extend(hops)
        counter += 1

    chain_df   = pd.DataFrame(all_chain_txns) if all_chain_txns else pd.DataFrame()
    wallets_df = reg.to_dataframe()

    # Aggregate totals into wallet records
    if not chain_df.empty and not wallets_df.empty:
        recv = chain_df.groupby('to_address')['net_amount'].sum()
        sent = chain_df.groupby('from_address')['amount'].sum()
        wallets_df['total_received'] = wallets_df['address'].map(recv).fillna(0)
        wallets_df['total_sent']     = wallets_df['address'].map(sent).fillna(0)

    return chain_df, wallets_df
