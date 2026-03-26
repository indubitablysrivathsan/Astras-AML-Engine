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
    'third_party_payments', 'round_tripping', 'smurfing'
]

HIGH_RISK_COUNTRIES = [
    'China', 'Russia', 'UAE', 'Panama', 'Cyprus', 'Cayman Islands',
    'British Virgin Islands', 'Seychelles', 'Belize', 'Malta',
    'Luxembourg', 'Mauritius', 'Hong Kong', 'Singapore', 'Switzerland'
]

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
