"""
Synthetic AML Data Generator (v2 - Realistic)
Generates realistic transaction data with:
- 10 money laundering typologies
- Gray-area customers that look suspicious but aren't (false positives)
- Noisy suspicious patterns mixed with normal activity
- Varied normal customers (international travelers, cash businesses, etc.)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import sqlite3
import random
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    NUM_CUSTOMERS, NUM_SUSPICIOUS, SIMULATION_DAYS, START_DATE_STR,
    TYPOLOGIES, DB_PATH, DATA_DIR,
    CRYPTO_RATES_USD, COUNTRY_CURRENCY_MAP,
    SUPPORTED_CRYPTO, CRYPTO_EXCHANGES,
)
from services.data_generation.rates import load_rates, get_usd_rate

fake = Faker('en_US')
np.random.seed(42)
random.seed(42)

START_DATE = datetime.strptime(START_DATE_STR, '%Y-%m-%d')

# Gray-area profiles: legitimate reasons for suspicious-looking behavior
GRAY_AREA_PROFILES = [
    'international_business',    # Legit business with lots of international wires
    'cash_restaurant',           # Restaurant owner with high cash deposits
    'freelancer_irregular',      # Freelancer with irregular large payments
    'real_estate_investor',      # Large wire transfers for property deals
    'travel_enthusiast',         # Frequent international transactions
    'family_remittance',         # Regular transfers to family overseas
    'seasonal_business',         # Seasonal business with burst patterns
    'day_trader',                # Frequent large deposits/withdrawals
    'crypto_investor',           # Legit crypto trader (false positive for crypto_laundering)
]

# Helper: resolve currency for a country
def _currency_for_country(country):
    return COUNTRY_CURRENCY_MAP.get(country, 'USD')


NUM_GRAY_AREA = 50  # ~5% of customers


def generate_customers(num_customers=NUM_CUSTOMERS, num_suspicious=NUM_SUSPICIOUS):
    print(f"Generating {num_customers:,} customer profiles...")
    print(f"  Including {NUM_GRAY_AREA} gray-area customers (expected false positives)")
    customers = []
    suspicious_indices = set(np.random.choice(num_customers, num_suspicious, replace=False))

    # Pick gray-area indices from non-suspicious customers
    remaining = [i for i in range(num_customers) if i not in suspicious_indices]
    gray_area_indices = set(np.random.choice(remaining, NUM_GRAY_AREA, replace=False))

    for i in range(num_customers):
        is_suspicious = i in suspicious_indices
        is_gray_area = i in gray_area_indices
        customer_type = np.random.choice(['individual', 'business'], p=[0.7, 0.3])

        if customer_type == 'individual':
            name = fake.name()
            if is_suspicious:
                occupation = np.random.choice(
                    ['Self-Employed', 'Consultant', 'Freelancer', 'Unemployed',
                     'Import/Export', 'Contractor', fake.job()])  # some have normal jobs
            elif is_gray_area:
                occupation = np.random.choice(
                    ['Restaurant Owner', 'Freelance Developer', 'Real Estate Agent',
                     'Day Trader', 'Travel Blogger', 'Import Consultant'])
            else:
                occupation = fake.job()
        else:
            name = fake.company()
            occupation = fake.bs()

        if is_suspicious and customer_type == 'business':
            annual_income = np.random.uniform(50000, 200000)
        elif is_gray_area:
            # Gray area customers often have higher incomes that justify some activity
            annual_income = np.random.uniform(80000, 500000)
        else:
            annual_income = np.clip(np.random.lognormal(np.log(75000), 0.8), 20000, 5000000)

        typology = np.random.choice(TYPOLOGIES) if is_suspicious else None
        gray_profile = np.random.choice(GRAY_AREA_PROFILES) if is_gray_area else None

        # Risk rating: suspicious=high, gray-area=medium, some normals=medium too
        if is_suspicious:
            risk_rating = np.random.choice(['high', 'medium'], p=[0.85, 0.15])
        elif is_gray_area:
            risk_rating = 'medium'
        else:
            risk_rating = np.random.choice(['low', 'medium'], p=[0.7, 0.3])

        # Crypto & international activity probabilities
        if is_suspicious and typology == 'crypto_laundering':
            intl_prob = np.random.uniform(0.6, 0.9)
            crypto_user = True
            preferred_crypto = np.random.choice(SUPPORTED_CRYPTO)
        elif is_gray_area and gray_profile == 'crypto_investor':
            intl_prob = np.random.uniform(0.2, 0.5)
            crypto_user = True
            preferred_crypto = np.random.choice(SUPPORTED_CRYPTO)
        elif is_gray_area and gray_profile in ('international_business', 'travel_enthusiast', 'family_remittance'):
            intl_prob = np.random.uniform(0.3, 0.7)
            crypto_user = np.random.random() < 0.08
            preferred_crypto = np.random.choice(SUPPORTED_CRYPTO) if crypto_user else None
        else:
            intl_prob = np.random.uniform(0.0, 0.15)
            crypto_user = np.random.random() < 0.05   # ~5% baseline adoption
            preferred_crypto = np.random.choice(SUPPORTED_CRYPTO) if crypto_user else None

        customers.append({
            'customer_id': i,
            'name': name,
            'customer_type': customer_type,
            'occupation': occupation,
            'annual_income': round(annual_income, 2),
            'address': fake.address().replace('\n', ', '),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip_code': fake.zipcode(),
            'ssn_ein': fake.ssn() if customer_type == 'individual' else fake.ein(),
            'account_open_date': (START_DATE - timedelta(days=np.random.randint(30, 3650))).strftime('%Y-%m-%d'),
            'risk_rating': risk_rating,
            'is_suspicious': is_suspicious,
            'typology': typology,
            'is_gray_area': is_gray_area,
            'gray_profile': gray_profile,
            'intl_activity_prob': round(intl_prob, 3),
            'crypto_user': crypto_user,
            'preferred_crypto': preferred_crypto,
        })

    return pd.DataFrame(customers)


def _make_txn(customer_id, date, txn_type, amount, method, location, description,
              counterparty=None, counterparty_account=None, counterparty_bank=None,
              country='USA', currency='USD', crypto_asset=None):
    # amount is stored in native currency units (EUR, INR, BTC, etc.).
    # usd_rate is the daily close rate from Yahoo Finance for that date,
    # stored as a reference column for downstream normalization.
    usd_rate = get_usd_rate(currency, date)
    return {
        'customer_id': customer_id,
        'transaction_date': date.strftime('%Y-%m-%d %H:%M:%S'),
        'transaction_type': txn_type,
        'amount': round(amount, 8) if crypto_asset else round(amount, 2),
        'currency': currency,
        'usd_rate': round(usd_rate, 8),
        'method': method,
        'location': location,
        'description': description,
        'counterparty': counterparty,
        'counterparty_account': counterparty_account,
        'counterparty_bank': counterparty_bank,
        'country': country,
        'crypto_asset': crypto_asset,
    }


# ══════════════════════════════════════════════════════════════════
# GRAY-AREA GENERATORS (look suspicious, but aren't)
# ══════════════════════════════════════════════════════════════════

def generate_international_business(customer, start_date):
    """Legit import/export business with lots of international wires."""
    txns = []
    partners = [(fake.company(), np.random.choice(['China', 'Germany', 'Japan', 'UK', 'India']))
                for _ in range(5)]
    for month in range(12):
        # Regular supplier payments (look like layering but are legit)
        for partner, country in partners[:np.random.randint(2, 5)]:
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 15))
            amt = np.random.uniform(15000, 80000)
            cur = _currency_for_country(country)
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal', amt, 'wire',
                                  'Wire Transfer', 'Supplier Payment', partner, fake.bban(),
                                  partner + ' Bank', country,
                                  currency=cur))
        # Revenue from clients
        for _ in range(np.random.randint(2, 6)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            amt = np.random.uniform(10000, 120000)
            cp = fake.company()
            rev_country = np.random.choice(['USA', 'UK', 'Germany'])
            rev_cur = _currency_for_country(rev_country)
            txns.append(_make_txn(customer['customer_id'], d, 'deposit', amt, 'wire',
                                  'Wire Transfer', 'Client Payment', cp, fake.bban(),
                                  cp + ' Bank', rev_country,
                                  currency=rev_cur))
    return txns


def generate_cash_restaurant(customer, start_date):
    """Restaurant with legitimately high cash deposits."""
    txns = []
    for month in range(12):
        # Daily cash deposits (looks like cash-intensive laundering)
        for day in range(np.random.randint(20, 28)):
            d = start_date + timedelta(days=month * 30 + day)
            # Revenue varies by day of week
            base = np.random.uniform(2000, 8000)
            if d.weekday() >= 4:  # weekends higher
                base *= 1.5
            txns.append(_make_txn(customer['customer_id'], d, 'deposit', base, 'cash',
                                  'Main Branch', 'Daily Business Deposit'))
        # Regular supplier payments
        for _ in range(np.random.randint(5, 10)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal',
                                  np.random.uniform(500, 5000),
                                  np.random.choice(['check', 'ach']),
                                  fake.company(), 'Supplier Payment', fake.company()))
    return txns


def generate_freelancer_irregular(customer, start_date):
    """Freelancer with large irregular payments (looks like structuring)."""
    txns = []
    for month in range(12):
        # 1-3 large project payments per month (irregular timing and amounts)
        num_projects = np.random.randint(0, 4)
        for _ in range(num_projects):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            # Some payments land just under 10K (coincidentally, not structuring)
            amt = np.random.choice([
                np.random.uniform(3000, 9500),   # smaller projects
                np.random.uniform(9000, 9900),   # looks like structuring!
                np.random.uniform(10000, 35000),  # larger projects
            ])
            txns.append(_make_txn(customer['customer_id'], d, 'deposit', amt,
                                  np.random.choice(['wire', 'ach']),
                                  'ACH Deposit', 'Freelance Payment', fake.company()))
        # Normal expenses
        for _ in range(np.random.randint(8, 15)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal',
                                  np.random.uniform(20, 1500), 'card',
                                  fake.company(), 'Purchase', fake.company()))
    return txns


def generate_real_estate_investor(customer, start_date):
    """Real estate investor with large, infrequent wire transfers."""
    txns = []
    # 2-4 property transactions per year (looks like round-tripping)
    num_deals = np.random.randint(2, 5)
    for deal in range(num_deals):
        d = start_date + timedelta(days=np.random.randint(0, 350))
        # Large outgoing payment for property
        amt = np.random.uniform(100000, 500000)
        txns.append(_make_txn(customer['customer_id'], d, 'withdrawal', amt, 'wire',
                              'Wire Transfer', 'Property Purchase', fake.company() + ' Realty',
                              fake.bban(), fake.company() + ' Bank'))
        # Rental income coming in monthly after purchase
        for m in range(np.random.randint(3, 10)):
            rd = d + timedelta(days=30 * (m + 1))
            if rd < start_date + timedelta(days=365):
                txns.append(_make_txn(customer['customer_id'], rd, 'deposit',
                                      np.random.uniform(2000, 8000), 'ach',
                                      'ACH Deposit', 'Rental Income', fake.name()))
    # Some international transfers (property abroad - looks suspicious)
    for _ in range(np.random.randint(1, 3)):
        d = start_date + timedelta(days=np.random.randint(0, 350))
        txns.append(_make_txn(customer['customer_id'], d, 'withdrawal',
                              np.random.uniform(50000, 200000), 'wire', 'Wire Transfer',
                              'International Property Investment', fake.company(),
                              fake.bban(), fake.company() + ' Bank',
                              np.random.choice(['Spain', 'Portugal', 'Mexico', 'Costa Rica'])))
    return txns


def generate_travel_enthusiast(customer, start_date):
    """Frequent traveler with many international transactions."""
    txns = []
    countries = ['France', 'Japan', 'Thailand', 'Brazil', 'UK', 'Italy',
                 'UAE', 'Singapore', 'Switzerland', 'Turkey']
    for month in range(12):
        # Normal income
        d = start_date + timedelta(days=month * 30 + 1)
        txns.append(_make_txn(customer['customer_id'], d, 'deposit',
                              customer['annual_income'] / 12, 'ach',
                              'ACH Deposit', 'Salary', fake.company()))
        # Travel spending (4-6 trips per year, cluster of intl transactions)
        if np.random.random() < 0.4:  # ~5 months with travel
            trip_country = np.random.choice(countries)
            trip_start = start_date + timedelta(days=month * 30 + np.random.randint(5, 20))
            for day in range(np.random.randint(3, 12)):
                td = trip_start + timedelta(days=day)
                txns.append(_make_txn(customer['customer_id'], td, 'withdrawal',
                                      np.random.uniform(50, 500), 'card',
                                      f'Hotel/Restaurant {trip_country}', 'Travel Expense',
                                      fake.company(), country=trip_country))
            # ATM withdrawal abroad
            txns.append(_make_txn(customer['customer_id'], trip_start, 'withdrawal',
                                  np.random.uniform(200, 1000), 'cash',
                                  f'ATM {trip_country}', 'ATM Withdrawal', country=trip_country))
    return txns


def generate_family_remittance(customer, start_date):
    """Regular remittances to family overseas (looks like rapid movement)."""
    txns = []
    family_country = np.random.choice(['Philippines', 'Mexico', 'India', 'Nigeria', 'Vietnam'])
    family_names = [fake.name() for _ in range(3)]
    rem_cur = _currency_for_country(family_country)
    for month in range(12):
        # Salary
        d = start_date + timedelta(days=month * 30 + 1)
        txns.append(_make_txn(customer['customer_id'], d, 'deposit',
                              customer['annual_income'] / 12, 'ach',
                              'ACH Deposit', 'Salary', fake.company()))
        # Monthly remittance (1-3 transfers)
        for fam in family_names[:np.random.randint(1, 3)]:
            rd = start_date + timedelta(days=month * 30 + np.random.randint(2, 10))
            txns.append(_make_txn(customer['customer_id'], rd, 'withdrawal',
                                  np.random.uniform(500, 3000), 'wire', 'Wire Transfer',
                                  'Family Support', fam, fake.bban(),
                                  fake.company() + ' Bank', family_country,
                                  currency=rem_cur))
        # Normal expenses
        for _ in range(np.random.randint(5, 12)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal',
                                  np.random.uniform(20, 800), 'card',
                                  fake.company(), 'Purchase'))
    return txns


def generate_seasonal_business(customer, start_date):
    """Seasonal business with burst patterns (looks like temporal bursts)."""
    txns = []
    # High season: months 5-8 (summer) and 11-12 (holidays)
    for month in range(12):
        is_peak = month in [4, 5, 6, 7, 10, 11]
        if is_peak:
            num_txns = np.random.randint(15, 30)  # burst of activity
            daily_rev = np.random.uniform(3000, 15000)
        else:
            num_txns = np.random.randint(3, 8)
            daily_rev = np.random.uniform(500, 3000)

        for _ in range(num_txns):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            txns.append(_make_txn(customer['customer_id'], d, 'deposit',
                                  daily_rev * np.random.uniform(0.5, 1.5),
                                  np.random.choice(['cash', 'card', 'ach']),
                                  'Main Branch', 'Business Revenue', fake.company()))
        # Expenses
        for _ in range(np.random.randint(3, 8)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal',
                                  np.random.uniform(200, 5000),
                                  np.random.choice(['check', 'ach']),
                                  fake.company(), 'Business Expense', fake.company()))
    return txns


def generate_day_trader(customer, start_date):
    """Day trader with rapid large movements (looks like rapid movement)."""
    txns = []
    brokerages = [fake.company() + ' Securities' for _ in range(3)]
    for month in range(12):
        # Funding transfers (large, rapid)
        for _ in range(np.random.randint(2, 6)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            amt = np.random.uniform(10000, 100000)
            broker = np.random.choice(brokerages)
            # Deposit from brokerage or send to brokerage
            if np.random.random() < 0.5:
                txns.append(_make_txn(customer['customer_id'], d, 'deposit', amt, 'wire',
                                      'Wire Transfer', 'Brokerage Transfer', broker,
                                      fake.bban(), broker))
            else:
                txns.append(_make_txn(customer['customer_id'], d, 'withdrawal', amt, 'wire',
                                      'Wire Transfer', 'Brokerage Funding', broker,
                                      fake.bban(), broker))
    return txns


def generate_legit_crypto_investor(customer, start_date):
    """Legitimate crypto trader — high volume but proportional to income.
    Creates false positives for crypto_laundering detection because the
    pattern (fiat -> exchange -> withdraw crypto gains) looks similar."""
    txns = []
    preferred = customer.get('preferred_crypto', 'BTC')
    exchange = np.random.choice(CRYPTO_EXCHANGES)
    exchange_acct = fake.bban()

    for month in range(12):
        # Salary / income deposit (USD, normal)
        sal_d = start_date + timedelta(days=month * 30 + 1)
        txns.append(_make_txn(customer['customer_id'], sal_d, 'deposit',
                              customer['annual_income'] / 12 * np.random.uniform(0.95, 1.05),
                              'ach', 'ACH Deposit', 'Salary', fake.company()))

        # 2-4 crypto purchases per month (looks like crypto_laundering but legit)
        # Amounts in crypto native units
        for _ in range(np.random.randint(2, 5)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(1, 28))
            if preferred == 'BTC':
                buy_amt = np.random.uniform(0.005, 0.12)
            elif preferred == 'ETH':
                buy_amt = np.random.uniform(0.1, 2.5)
            else:  # USDT
                buy_amt = np.random.uniform(500, 8000)
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal', buy_amt,
                                  'crypto_exchange', exchange, f'Buy {preferred}',
                                  exchange, exchange_acct, exchange,
                                  currency=preferred, crypto_asset=preferred))

        # Occasional crypto sale (profit taking — deposit back in USD)
        if np.random.random() < 0.35:
            d = start_date + timedelta(days=month * 30 + np.random.randint(15, 28))
            sell_amt = np.random.uniform(2000, 15000)  # USD proceeds
            txns.append(_make_txn(customer['customer_id'], d, 'deposit', sell_amt,
                                  'crypto_exchange', exchange, f'Sell {preferred}',
                                  exchange, exchange_acct, exchange,
                                  currency='USD', crypto_asset=preferred))

        # Normal card spending
        for _ in range(np.random.randint(5, 12)):
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal',
                                  np.random.uniform(20, 800), 'card',
                                  fake.company(), 'Purchase'))
    return txns


GRAY_AREA_GENERATORS = {
    'international_business': generate_international_business,
    'cash_restaurant': generate_cash_restaurant,
    'freelancer_irregular': generate_freelancer_irregular,
    'real_estate_investor': generate_real_estate_investor,
    'travel_enthusiast': generate_travel_enthusiast,
    'family_remittance': generate_family_remittance,
    'seasonal_business': generate_seasonal_business,
    'day_trader': generate_day_trader,
    'crypto_investor': generate_legit_crypto_investor,
}


# ══════════════════════════════════════════════════════════════════
# SUSPICIOUS PATTERN GENERATORS (with added noise)
# ══════════════════════════════════════════════════════════════════

def generate_structuring_pattern(customer, start_date, num_days=30):
    txns = []
    # Fewer deposits than before, some above 10K (imperfect structuring)
    num_deposits = np.random.randint(8, 20)
    for _ in range(num_deposits):
        # 80% below 10K, 20% above (sloppy structuring)
        if np.random.random() < 0.8:
            amount = np.random.uniform(7000, 9900)
        else:
            amount = np.random.uniform(10000, 15000)
        d = start_date + timedelta(days=np.random.randint(0, num_days))
        branch = np.random.choice(['Downtown Branch', 'Westside Branch', 'Airport Branch',
                                   'Suburban Branch', 'Mall Branch'])
        txns.append(_make_txn(customer['customer_id'], d, 'deposit', amount, 'cash', branch, 'Cash Deposit'))
    return txns


def generate_rapid_movement_pattern(customer, start_date, num_cycles=5):
    txns = []
    # Bursty: small pool of counterparties reused, occasional new ones
    src_pool = [(fake.company(), fake.bban()) for _ in range(3)]
    dst_pool = [(fake.name(), fake.bban(), fake.company() + ' Bank') for _ in range(2)]
    for cycle in range(num_cycles):
        cs = start_date + timedelta(days=cycle * np.random.randint(20, 45))
        dep_amt = np.random.uniform(15000, 150000)
        # Burst: reuse same counterparty for consecutive cycles, expand occasionally
        if cycle > 2 and np.random.random() < 0.3:
            src_pool.append((fake.company(), fake.bban()))
        cp, cp_acct = src_pool[cycle % len(src_pool)]
        txns.append(_make_txn(customer['customer_id'], cs, 'deposit', dep_amt, 'wire', 'Wire Transfer',
                              'Incoming Wire', cp, cp_acct, cp + ' Bank',
                              np.random.choice(['China', 'Russia', 'UAE', 'Panama', 'Cyprus', 'USA', 'UK'])))
        delay_hrs = int(np.random.choice([np.random.randint(12, 48),
                                           np.random.randint(48, 240)]))
        wd = cs + timedelta(hours=delay_hrs)
        wa = dep_amt * np.random.uniform(0.85, 0.99)
        dst = dst_pool[cycle % len(dst_pool)]
        txns.append(_make_txn(customer['customer_id'], wd, 'withdrawal', wa, 'wire', 'Wire Transfer',
                              'Outgoing Wire', dst[0], dst[1], dst[2],
                              np.random.choice(['Cayman Islands', 'Switzerland', 'Singapore', 'Hong Kong', 'USA'])))
    return txns


def generate_cash_intensive_pattern(customer, start_date, num_months=12):
    txns = []
    benchmark = customer['annual_income'] / 12 * 0.6
    # Multiplier varies: some months more suspicious than others
    for month in range(num_months):
        multiplier = np.random.uniform(1.5, 5.0)  # wider range, some months look normal
        monthly_cash = benchmark * multiplier
        num_dep = np.random.randint(5, 13)
        for _ in range(num_dep):
            amount = monthly_cash / num_dep * np.random.uniform(0.6, 1.4)
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 30))
            txns.append(_make_txn(customer['customer_id'], d, 'deposit', amount, 'cash',
                                  'Main Branch', f'Business Cash Deposit - {customer["name"]}'))
    return txns


def generate_shell_company_pattern(customer, start_date, num_wires=20):
    txns = []
    num_wires = np.random.randint(10, 20)
    # Bursty: 2-3 shell entities feed into 1-2 destinations, reused heavily
    shell_src = [(fake.company() + ' Ltd', fake.bban()) for _ in range(np.random.randint(2, 4))]
    shell_dst = [(fake.company() + ' Holdings', fake.bban()) for _ in range(np.random.randint(1, 3))]
    for i in range(num_wires):
        d = start_date + timedelta(days=np.random.randint(0, 365))
        amt_in = np.random.uniform(50000, 500000)
        src = shell_src[i % len(shell_src)]
        in_country = np.random.choice(
            ['British Virgin Islands', 'Seychelles', 'Belize', 'Malta', 'USA', 'UK', 'USA'],
        )
        txns.append(_make_txn(customer['customer_id'], d, 'deposit', amt_in, 'wire', 'Wire Transfer',
                              'International Wire - Consulting Fees', src[0], src[1], src[0] + ' Bank',
                              in_country))
        wd = d + timedelta(days=np.random.randint(2, 15))
        amt_out = amt_in * np.random.uniform(0.85, 0.98)
        dst = shell_dst[i % len(shell_dst)]
        txns.append(_make_txn(customer['customer_id'], wd, 'withdrawal', amt_out, 'wire', 'Wire Transfer',
                              'Wire Transfer - Vendor Payment', dst[0], dst[1], dst[0] + ' Bank',
                              np.random.choice(['Luxembourg', 'Panama', 'Cyprus', 'Mauritius', 'USA'])))
        # Inject new shell entity mid-sequence (expansion burst)
        if i == num_wires // 2:
            shell_src.append((fake.company() + ' Intl', fake.bban()))
    return txns


def generate_smurfing_pattern(customer, start_date, num_events=10):
    txns = []
    num_smurfs = np.random.randint(3, 6)
    smurf_names = [fake.name() for _ in range(num_smurfs)]
    # Fewer events, not all smurfs deposit every time
    for event in range(np.random.randint(5, num_events)):
        ed = start_date + timedelta(days=event * np.random.randint(10, 25))
        active_smurfs = np.random.choice(smurf_names,
                                          size=np.random.randint(2, len(smurf_names) + 1),
                                          replace=False)
        for smurf in active_smurfs:
            amount = np.random.uniform(2000, 9500)
            dt = ed + timedelta(hours=np.random.randint(0, 48))
            branch = np.random.choice(['North Branch', 'South Branch', 'East Branch',
                                       'West Branch', 'Central Branch', 'Airport Branch'])
            txns.append(_make_txn(customer['customer_id'], dt, 'deposit', amount, 'cash', branch,
                                  f'Cash Deposit by {smurf}', smurf))
    return txns


def generate_third_party_pattern(customer, start_date, num_payments=15):
    txns = []
    parties = [fake.name() for _ in range(8)]
    num_payments = np.random.randint(8, 15)
    for _ in range(num_payments):
        pd_ = start_date + timedelta(days=np.random.randint(0, 365))
        recv = np.random.uniform(3000, 25000)
        cp = np.random.choice(parties)
        txns.append(_make_txn(customer['customer_id'], pd_, 'deposit', recv, 'wire', 'Wire Transfer',
                              'Third Party Transfer', cp, fake.bban(), fake.company() + ' Bank'))
        pay_d = pd_ + timedelta(days=np.random.randint(1, 10))
        pay_a = recv * np.random.uniform(0.75, 0.95)
        cp2 = np.random.choice(parties)
        txns.append(_make_txn(customer['customer_id'], pay_d, 'withdrawal', pay_a, 'wire', 'Wire Transfer',
                              'Payment to Third Party', cp2, fake.bban(), fake.company() + ' Bank'))
    return txns


def generate_layering_pattern(customer, start_date, num_chains=5):
    txns = []
    num_chains = np.random.randint(3, 6)
    # Layering reuses a small set of intermediaries across chains (bursty)
    intermediaries = [(fake.company(), fake.bban()) for _ in range(4)]
    for chain in range(num_chains):
        cs = start_date + timedelta(days=chain * np.random.randint(40, 80))
        initial = np.random.uniform(30000, 200000)
        entry = intermediaries[chain % len(intermediaries)]
        txns.append(_make_txn(customer['customer_id'], cs, 'deposit', initial, 'wire', 'Wire Transfer',
                              'Wire Transfer In', entry[0], entry[1], entry[0] + ' Bank'))
        cur_amt, cur_date = initial, cs
        for hop in range(np.random.randint(3, 8)):
            cur_date += timedelta(days=np.random.randint(2, 15))
            cur_amt *= np.random.uniform(0.90, 0.99)
            t_type = 'withdrawal' if hop % 2 == 1 else 'deposit'
            # Reuse intermediary pool — same entities seen across hops
            cp_h = intermediaries[(chain + hop) % len(intermediaries)]
            txns.append(_make_txn(customer['customer_id'], cur_date, t_type, cur_amt, 'wire', 'Wire Transfer',
                                  f'Transfer {hop + 1}', cp_h[0], cp_h[1], cp_h[0] + ' Bank',
                                  np.random.choice(['USA', 'UK', 'Germany', 'Singapore', 'Switzerland'])))
        # Expansion burst: inject new intermediary every other chain
        if chain % 2 == 1:
            intermediaries.append((fake.company(), fake.bban()))
    return txns


def generate_round_tripping_pattern(customer, start_date, num_cycles=4):
    txns = []
    num_cycles = np.random.randint(2, 5)
    # Round-tripping: same entity pair used across cycles (bursty concentration)
    outbound_entity = (fake.company(), fake.bban(), fake.company() + ' Bank')
    # Return entity is a slightly different name variant (same beneficial owner pattern)
    inbound_entity = (outbound_entity[0] + ' Holdings Ltd', fake.bban(), fake.company() + ' Bank')
    for cycle in range(num_cycles):
        cs = start_date + timedelta(days=cycle * np.random.randint(60, 120))
        amount = np.random.uniform(30000, 300000)
        txns.append(_make_txn(customer['customer_id'], cs, 'withdrawal', amount, 'wire', 'Wire Transfer',
                              'Investment Wire - Overseas', outbound_entity[0], outbound_entity[1],
                              outbound_entity[2],
                              np.random.choice(['Cayman Islands', 'British Virgin Islands', 'Panama', 'USA'])))
        ret_d = cs + timedelta(days=np.random.randint(20, 90))
        ret_a = amount * np.random.uniform(0.98, 1.15)
        txns.append(_make_txn(customer['customer_id'], ret_d, 'deposit', ret_a, 'wire', 'Wire Transfer',
                              'Investment Return - Foreign Entity', inbound_entity[0], inbound_entity[1],
                              inbound_entity[2],
                              np.random.choice(['Luxembourg', 'Switzerland', 'Singapore', 'USA'])))
    return txns


def generate_funnel_account_pattern(customer, start_date, num_cycles=8):
    txns = []
    sources = [fake.name() for _ in range(6)]
    beneficiaries = [fake.name() for _ in range(2)]
    num_cycles = np.random.randint(4, 9)
    for cycle in range(num_cycles):
        cs = start_date + timedelta(days=cycle * np.random.randint(30, 60))
        total = 0
        # Not all sources every cycle
        active = np.random.choice(sources, size=np.random.randint(3, len(sources) + 1), replace=False)
        for src in active:
            amt = np.random.uniform(3000, 20000)
            total += amt
            dd = cs + timedelta(days=np.random.randint(0, 7))
            txns.append(_make_txn(customer['customer_id'], dd, 'deposit', amt, 'wire', 'Wire Transfer',
                                  'Transfer from Associate', src, fake.bban(), fake.company() + ' Bank'))
        remaining = total * np.random.uniform(0.80, 0.95)
        for ben in beneficiaries:
            sa = remaining / len(beneficiaries) * np.random.uniform(0.8, 1.2)
            sd = cs + timedelta(days=np.random.randint(3, 12))
            txns.append(_make_txn(customer['customer_id'], sd, 'withdrawal', sa, 'wire', 'Wire Transfer',
                                  'Distribution Payment', ben, fake.bban(), fake.company() + ' Bank'))
    return txns


def generate_trade_based_pattern(customer, start_date, num_invoices=8):
    txns = []
    goods = ['Electronics', 'Textiles', 'Auto Parts', 'Machinery', 'Pharmaceuticals']
    num_invoices = np.random.randint(4, 9)
    # Bursty: same 2 trade partners used repeatedly (concentrated risk)
    buyer = (fake.company() + ' Trading Co', fake.bban())
    supplier = (fake.company() + ' Mfg', fake.bban())
    for i in range(num_invoices):
        id_ = start_date + timedelta(days=np.random.randint(0, 365))
        fair = np.random.uniform(10000, 50000)
        invoiced = fair * np.random.uniform(1.5, 4.0)
        txns.append(_make_txn(customer['customer_id'], id_, 'deposit', invoiced, 'wire', 'Wire Transfer',
                              f'Trade Payment - {np.random.choice(goods)}', buyer[0], buyer[1],
                              buyer[0] + ' Bank',
                              np.random.choice(['China', 'Turkey', 'India', 'Nigeria', 'Vietnam', 'USA'])))
        sd = id_ + timedelta(days=np.random.randint(5, 20))
        sa = fair * np.random.uniform(0.7, 1.1)
        txns.append(_make_txn(customer['customer_id'], sd, 'withdrawal', sa, 'wire', 'Wire Transfer',
                              f'Supplier Payment - {np.random.choice(goods)}', supplier[0], supplier[1],
                              supplier[0] + ' Bank',
                              np.random.choice(['China', 'Turkey', 'India', 'USA'])))
        # Expansion burst halfway through
        if i == num_invoices // 2:
            buyer = (fake.company() + ' Imports', fake.bban())
    return txns


def generate_normal_transactions(customer, start_date, num_days=365):
    """Generate normal banking activity - now with more variety."""
    txns = []
    monthly_income = customer['annual_income'] / 12
    for month in range(12):
        sal_d = start_date + timedelta(days=month * 30 + 1)
        txns.append(_make_txn(customer['customer_id'], sal_d, 'deposit',
                              monthly_income * np.random.uniform(0.95, 1.05),
                              'ach' if customer['customer_type'] == 'individual' else 'wire',
                              'ACH Deposit',
                              'Salary' if customer['customer_type'] == 'individual' else 'Revenue',
                              fake.company()))
        # Random expenses
        for _ in range(np.random.randint(5, 16)):
            exp_d = start_date + timedelta(days=month * 30 + np.random.randint(0, 30))
            txns.append(_make_txn(customer['customer_id'], exp_d, 'withdrawal',
                                  np.random.uniform(50, 2000),
                                  np.random.choice(['card', 'ach', 'check']),
                                  fake.company(),
                                  np.random.choice(['Purchase', 'Bill Payment', 'ATM Withdrawal']),
                                  fake.company()))

        # Occasional wire/intl for some normal customers (adds noise)
        if np.random.random() < 0.05:  # 5% chance per month
            d = start_date + timedelta(days=month * 30 + np.random.randint(0, 28))
            txns.append(_make_txn(customer['customer_id'], d, 'withdrawal',
                                  np.random.uniform(500, 5000), 'wire', 'Wire Transfer',
                                  'Personal Transfer', fake.name(), fake.bban(),
                                  fake.company() + ' Bank',
                                  np.random.choice(['USA', 'Mexico', 'Canada', 'UK'])))
    return txns


def generate_crypto_laundering_pattern(customer, start_date, num_cycles=5):
    """Cryptocurrency-facilitated money laundering: fiat -> crypto exchange ->
    cross-border transfer -> conversion back to different fiat currency.
    Each cycle follows the 4-stage placement-layering-integration pattern
    through digital asset rails.

    Counterparties are BURSTY: a small pool of shell entities is reused
    across cycles (triggers counterparty-concentration signals), with
    occasional new entities injected mid-run (triggers expansion signals).
    """
    txns = []
    preferred = customer.get('preferred_crypto', np.random.choice(SUPPORTED_CRYPTO))
    num_cycles = np.random.randint(3, 7)

    # Crypto amount ranges per asset (realistic native-currency units)
    CRYPTO_AMT_RANGES = {
        'BTC': (0.05, 3.0),      # $3,250 – $195,000 at 65k
        'ETH': (0.5, 50.0),      # $1,700 – $170,000 at 3.4k
        'USDT': (5000, 200000),   # stablecoin, 1:1 USD
    }

    # Fiat amount ranges per currency (in local currency units)
    FIAT_AMT_RANGES = {
        'USD': (20000, 200000),
        'EUR': (18000, 180000),
        'AED': (75000, 750000),
        'INR': (1500000, 15000000),
    }

    # Country/currency pairs for the laundering corridors
    corridors = [
        ('UAE', 'AED'), ('India', 'INR'), ('Germany', 'EUR'),
        ('Singapore', 'USD'), ('UK', 'EUR'), ('Switzerland', 'EUR'),
    ]

    # --- BURSTY counterparty pool ---
    # Small fixed pool (3-4 entities) reused heavily → triggers concentration
    shell_pool = [(fake.company(), fake.bban(), fake.company() + ' Bank')
                  for _ in range(np.random.randint(3, 5))]
    exchange_pool = list(np.random.choice(CRYPTO_EXCHANGES, size=2, replace=False))

    for cycle in range(num_cycles):
        cs = start_date + timedelta(days=cycle * np.random.randint(30, 70))

        # Occasionally inject a NEW counterparty mid-run (expansion burst)
        if cycle > 0 and np.random.random() < 0.3:
            shell_pool.append((fake.company(), fake.bban(), fake.company() + ' Bank'))

        # Pick from the small pool (bursty reuse)
        shell = shell_pool[cycle % len(shell_pool)]
        exchange = exchange_pool[cycle % len(exchange_pool)]

        # --- Stage 1: Fiat deposit (placement) ---
        src_country, src_currency = corridors[cycle % len(corridors)]
        fiat_lo, fiat_hi = FIAT_AMT_RANGES.get(src_currency, (20000, 200000))
        fiat_amt = np.random.uniform(fiat_lo, fiat_hi)
        txns.append(_make_txn(customer['customer_id'], cs, 'deposit',
                              fiat_amt, 'wire', 'Wire Transfer', 'Incoming Wire',
                              shell[0], shell[1], shell[2], src_country,
                              currency=src_currency))

        # --- Stage 2: Crypto purchase at exchange (layering entry) ---
        buy_delay = timedelta(hours=np.random.randint(2, 48))
        crypto_lo, crypto_hi = CRYPTO_AMT_RANGES.get(preferred, (0.1, 5.0))
        crypto_amt = np.random.uniform(crypto_lo, crypto_hi)
        txns.append(_make_txn(customer['customer_id'], cs + buy_delay, 'withdrawal',
                              crypto_amt, 'crypto_exchange', exchange,
                              f'Purchase {preferred}', exchange, fake.bban(), exchange,
                              country=src_country, currency=preferred,
                              crypto_asset=preferred))

        # --- Stage 3: Cross-border crypto deposit (different country/exchange) ---
        xfer_delay = timedelta(days=np.random.randint(1, 14))
        dest_country, dest_currency = corridors[(cycle + 2) % len(corridors)]
        recv_exchange = exchange_pool[(cycle + 1) % len(exchange_pool)]
        # Received crypto amount slightly less (network fees)
        recv_crypto = crypto_amt * np.random.uniform(0.97, 0.999)
        txns.append(_make_txn(customer['customer_id'], cs + buy_delay + xfer_delay,
                              'deposit', recv_crypto, 'crypto_exchange', recv_exchange,
                              f'Receive {preferred}', recv_exchange,
                              fake.bban(), recv_exchange, dest_country,
                              currency=preferred, crypto_asset=preferred))

        # --- Stage 4: Fiat withdrawal to bank (integration) ---
        cash_out_delay = timedelta(days=np.random.randint(1, 7))
        dest_lo, dest_hi = FIAT_AMT_RANGES.get(dest_currency, (20000, 200000))
        # Cash out in destination fiat (slightly less than deposited value)
        final_fiat = np.random.uniform(dest_lo * 0.7, dest_hi * 0.6)
        dest_shell = shell_pool[(cycle + 1) % len(shell_pool)]
        txns.append(_make_txn(customer['customer_id'],
                              cs + buy_delay + xfer_delay + cash_out_delay,
                              'withdrawal', final_fiat, 'wire', 'Wire Transfer',
                              f'Funds Transfer', dest_shell[0],
                              dest_shell[1], dest_shell[2], dest_country,
                              currency=dest_currency))

    return txns


PATTERN_GENERATORS = {
    'structuring': generate_structuring_pattern,
    'rapid_movement': generate_rapid_movement_pattern,
    'cash_intensive': generate_cash_intensive_pattern,
    'shell_company': generate_shell_company_pattern,
    'smurfing': generate_smurfing_pattern,
    'third_party_payments': generate_third_party_pattern,
    'layering': generate_layering_pattern,
    'round_tripping': generate_round_tripping_pattern,
    'funnel_account': generate_funnel_account_pattern,
    'trade_based': generate_trade_based_pattern,
    'crypto_laundering': generate_crypto_laundering_pattern,
}


def generate_all_transactions(customers_df, start_date=START_DATE):
    print("\nGenerating transactions...")
    all_txns = []
    for idx, customer in customers_df.iterrows():
        if customer['is_suspicious']:
            gen = PATTERN_GENERATORS.get(customer['typology'], generate_structuring_pattern)
            suspicious_txns = gen(customer, start_date)
            normal_txns = generate_normal_transactions(customer, start_date)
            # Mix MORE normal transactions in (50-80% of total, not just 20)
            num_normal = int(len(suspicious_txns) * np.random.uniform(0.5, 1.5))
            all_txns.extend(suspicious_txns)
            all_txns.extend(normal_txns[:num_normal])
        elif customer.get('is_gray_area', False):
            # Gray-area: generate their special pattern + normal activity
            profile = customer.get('gray_profile', 'cash_restaurant')
            gen = GRAY_AREA_GENERATORS.get(profile, generate_cash_restaurant)
            all_txns.extend(gen(customer, start_date))
            # Also add some normal transactions
            all_txns.extend(generate_normal_transactions(customer, start_date)[:30])
        else:
            all_txns.extend(generate_normal_transactions(customer, start_date))
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1:,} / {len(customers_df):,} customers")
    return pd.DataFrame(all_txns)


def create_alerts(customers_df, transactions_df):
    """Create alerts for suspicious AND gray-area customers (false positives expected)."""
    print("\nGenerating alerts...")
    alerts = []
    alert_id = 0

    # Alerts for suspicious customers
    suspicious = customers_df[customers_df['is_suspicious']]
    for _, customer in suspicious.iterrows():
        ctxns = transactions_df[transactions_df['customer_id'] == customer['customer_id']]
        if len(ctxns) == 0:
            continue
        alerts.append({
            'alert_id': alert_id,
            'customer_id': customer['customer_id'],
            'alert_date': ctxns['transaction_date'].max(),
            'alert_type': customer['typology'],
            'severity': np.random.choice(['high', 'medium'], p=[0.8, 0.2]),
            'total_amount': round(ctxns['amount'].sum(), 2),
            'num_transactions': len(ctxns),
            'avg_transaction_amount': round(ctxns['amount'].mean(), 2),
            'status': 'open',
            'assigned_analyst': fake.name(),
            'sar_filed': False
        })
        alert_id += 1

    # Alerts for some gray-area customers (these are the false positives)
    gray_area = customers_df[customers_df.get('is_gray_area', False) == True]
    # ~60% of gray-area customers trigger alerts
    alerted_gray = gray_area.sample(frac=0.6, random_state=42) if len(gray_area) > 0 else gray_area
    for _, customer in alerted_gray.iterrows():
        ctxns = transactions_df[transactions_df['customer_id'] == customer['customer_id']]
        if len(ctxns) == 0:
            continue
        # Map gray profile to a typology it resembles
        profile_to_typology = {
            'international_business': 'layering',
            'cash_restaurant': 'cash_intensive',
            'freelancer_irregular': 'structuring',
            'real_estate_investor': 'round_tripping',
            'travel_enthusiast': 'rapid_movement',
            'family_remittance': 'rapid_movement',
            'seasonal_business': 'cash_intensive',
            'day_trader': 'rapid_movement',
            'crypto_investor': 'crypto_laundering',
        }
        apparent_typology = profile_to_typology.get(customer.get('gray_profile', ''), 'structuring')
        alerts.append({
            'alert_id': alert_id,
            'customer_id': customer['customer_id'],
            'alert_date': ctxns['transaction_date'].max(),
            'alert_type': apparent_typology,
            'severity': np.random.choice(['high', 'medium', 'low'], p=[0.2, 0.5, 0.3]),
            'total_amount': round(ctxns['amount'].sum(), 2),
            'num_transactions': len(ctxns),
            'avg_transaction_amount': round(ctxns['amount'].mean(), 2),
            'status': 'open',
            'assigned_analyst': fake.name(),
            'sar_filed': False
        })
        alert_id += 1

    print(f"  Suspicious alerts: {len(suspicious)}")
    print(f"  Gray-area alerts (false positives): {len(alerted_gray)}")
    return pd.DataFrame(alerts)


def save_to_database(customers_df, transactions_df, alerts_df, db_path=DB_PATH):
    print(f"\nSaving to {db_path}")
    conn = sqlite3.connect(db_path)
    customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
    alerts_df.to_sql('alerts', conn, if_exists='replace', index=False)
    conn.close()
    print(f"  Customers: {len(customers_df):,} | Transactions: {len(transactions_df):,} | Alerts: {len(alerts_df):,}")


def run():
    print("=" * 60)
    print("AML SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Pre-fetch / load daily FX + crypto rates for simulation period
    load_rates()

    customers_df = generate_customers()
    transactions_df = generate_all_transactions(customers_df)
    transactions_df['transaction_id'] = range(len(transactions_df))
    transactions_df = transactions_df.sort_values(['customer_id', 'transaction_date'])
    alerts_df = create_alerts(customers_df, transactions_df)
    save_to_database(customers_df, transactions_df, alerts_df)

    os.makedirs(os.path.join(DATA_DIR, 'generated'), exist_ok=True)
    customers_df.to_csv(os.path.join(DATA_DIR, 'generated', 'customers.csv'), index=False)
    transactions_df.to_csv(os.path.join(DATA_DIR, 'generated', 'transactions.csv'), index=False)
    alerts_df.to_csv(os.path.join(DATA_DIR, 'generated', 'alerts.csv'), index=False)

    print(f"\nTotal customers: {len(customers_df):,}")
    print(f"  Suspicious: {customers_df['is_suspicious'].sum():,}")
    print(f"  Gray-area: {customers_df.get('is_gray_area', pd.Series(dtype=bool)).sum()}")
    print(f"  Normal: {len(customers_df) - customers_df['is_suspicious'].sum() - customers_df.get('is_gray_area', pd.Series(dtype=bool)).sum()}")
    print(f"Transactions: {len(transactions_df):,}")
    print(f"Alerts: {len(alerts_df):,}")
    return customers_df, transactions_df, alerts_df


if __name__ == "__main__":
    run()
