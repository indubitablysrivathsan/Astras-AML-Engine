"""ASTRAS Configuration"""
import os


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'aml_data.db')
CHROMA_DIR = os.path.join(BASE_DIR, 'chroma_db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Data generation
NUM_CUSTOMERS = 1000
NUM_SUSPICIOUS = 100
SIMULATION_DAYS = 365
START_DATE_STR = '2025-01-01'

TYPOLOGIES = [
    'structuring', 'rapid_movement', 'layering', 'trade_based',
    'cash_intensive', 'shell_company', 'funnel_account',
    'third_party_payments', 'round_tripping', 'smurfing',
    'crypto_laundering'
]

HIGH_RISK_COUNTRIES = [
    'China', 'Russia', 'UAE', 'Panama', 'Cyprus', 'Cayman Islands',
    'British Virgin Islands', 'Seychelles', 'Belize', 'Malta',
    'Luxembourg', 'Mauritius', 'Hong Kong', 'Singapore', 'Switzerland'
]

# Currency & Cryptocurrency
SUPPORTED_CURRENCIES = ['USD', 'EUR', 'AED', 'INR']
SUPPORTED_CRYPTO = ['BTC', 'ETH', 'USDT']

FX_RATES = {          # to-USD multipliers (amount * rate = USD equivalent)
    'USD': 1.0,
    'EUR': 1.09,      # 1 EUR = 1.09 USD
    'AED': 0.2723,    # 1 AED = 0.27 USD
    'INR': 0.01198,   # 1 INR = 0.012 USD
}

CRYPTO_RATES_USD = {   # 1 unit of crypto in USD
    'BTC': 65000.0,
    'ETH': 3400.0,
    'USDT': 1.0,
}

# PPP price-level index per country (World Bank 2024, USA = 1.0)
# PLI = PPP_rate / FX_rate — how cheap a country is relative to the US.
# Low PLI = cheap country.  High PLI = expensive country.
# Used to compute transaction asymmetry: log(PLI_receiver / PLI_sender)
# High positive asymmetry = cheap→expensive flow = economically suspicious.
COUNTRY_PPL = {
    'USA':                  1.00,
    'UK':                   0.82,
    'Germany':              0.80,
    'France':               0.79,
    'Italy':                0.72,
    'Spain':                0.68,
    'Portugal':             0.63,
    'Switzerland':          1.10,   # more expensive than US
    'UAE':                  0.50,
    'Singapore':            0.84,
    'Hong Kong':            0.81,
    'Japan':                0.65,
    'China':                0.46,
    'India':                0.27,
    'Philippines':          0.30,
    'Vietnam':              0.29,
    'Nigeria':              0.28,
    'Mexico':               0.40,
    'Panama':               0.55,
    'Russia':               0.38,
    'Turkey':               0.32,
    # Offshore/secrecy jurisdictions — treat as opaque (use 0.6 neutral)
    'Cayman Islands':       0.60,
    'British Virgin Islands': 0.60,
    'Belize':               0.60,
    'Seychelles':           0.60,
    'Luxembourg':           0.85,
    'Malta':                0.70,
    'Mauritius':            0.45,
    'Cyprus':               0.68,
    'Canada':               0.87,
}
# Fallback for unmapped countries
DEFAULT_PPL = 0.60

COUNTRY_CURRENCY_MAP = {
    'Germany': 'EUR', 'France': 'EUR', 'Italy': 'EUR', 'Spain': 'EUR',
    'UK': 'EUR', 'Portugal': 'EUR', 'Switzerland': 'EUR',
    'UAE': 'AED',
    'India': 'INR',
    'Philippines': 'INR',   # remittance corridor, often settles in INR
}
# All unmapped countries default to 'USD'

CRYPTO_EXCHANGES = [
    'Binance', 'Coinbase', 'Kraken', 'OKX',
    'Bybit', 'Gemini', 'KuCoin', 'Huobi', 'Bitfinex', 'Bitstamp',
]

# Exchange risk profiles (used by crypto_chain.py and crypto_flow.py)
EXCHANGE_PROFILES = {
    'Binance':    {'risk_score': 0.55, 'kyc_level': 'medium', 'jurisdiction': 'Cayman Islands'},
    'Coinbase':   {'risk_score': 0.15, 'kyc_level': 'high',   'jurisdiction': 'USA'},
    'Kraken':     {'risk_score': 0.20, 'kyc_level': 'high',   'jurisdiction': 'USA'},
    'OKX':        {'risk_score': 0.65, 'kyc_level': 'low',    'jurisdiction': 'Seychelles'},
    'Bybit':      {'risk_score': 0.60, 'kyc_level': 'low',    'jurisdiction': 'UAE'},
    'Gemini':     {'risk_score': 0.10, 'kyc_level': 'high',   'jurisdiction': 'USA'},
    'KuCoin':     {'risk_score': 0.70, 'kyc_level': 'low',    'jurisdiction': 'Seychelles'},
    'Huobi':      {'risk_score': 0.65, 'kyc_level': 'medium', 'jurisdiction': 'Seychelles'},
    'Bitfinex':   {'risk_score': 0.50, 'kyc_level': 'medium', 'jurisdiction': 'British Virgin Islands'},
    'Bitstamp':   {'risk_score': 0.15, 'kyc_level': 'high',   'jurisdiction': 'Luxembourg'},
}

# Behavioral signals
ROLLING_WINDOW_DAYS = 30
ENTROPY_THRESHOLD = 0.5
BURST_THRESHOLD = 3.0
COUNTERPARTY_EXPANSION_THRESHOLD = 5

# BSI
BSI_CRITICAL = 25
BSI_HIGH_DRIFT = 50
BSI_MODERATE = 75
BSI_STABLE = 76

# Adaptive monitoring
MONITORING_LEVELS = {
    'standard': {'check_interval_days': 30, 'anomaly_threshold': 0.7},
    'enhanced': {'check_interval_days': 7, 'anomaly_threshold': 0.5},
    'intensive': {'check_interval_days': 1, 'anomaly_threshold': 0.3},
    'immediate': {'check_interval_days': 0, 'anomaly_threshold': 0.1},
}

# XGBoost
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'auc',
}

# LLM
LLM_MODEL = 'mistral:7b'
EMBEDDING_MODEL = 'nomic-embed-text'
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2000

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.7
