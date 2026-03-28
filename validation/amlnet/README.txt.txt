AMLNet - Synthetic Anti-Money Laundering Transaction Dataset

DESCRIPTION:
This dataset contains over 1 million synthetic financial transactions (1,090,173) generated using the AMLNet framework for anti-money laundering research.

CONTENTS:
- 1,090,173 total transactions across multiple categories
- 1,745 labeled money laundering transactions (~0.16%)
- 195-day simulation period
- AUSTRAC-compliant suspicious patterns

DATA FORMAT:
CSV file with 17 columns containing transaction details.

CORE TRANSACTION DATA:
- step: Sequential transaction step/ID
- type: Payment method (TRANSFER, OSKO, BPAY, EFTPOS, DEBIT, NPP)
- amount: Transaction amount in Australian Dollar 
- category: 11 Transaction categories (Education, Housing, Food, Healthcare, Transport, Recreation,    Cryptocurrency, Property Investment, Shell Company, Utilities, Other)
- nameOrig: Originating customer ID (e.g., C3511)
- nameDest: Destination customer/merchant ID (e.g., C4945, M558)
- oldbalanceOrg: Account balance before transaction
- newbalanceOrig: Account balance after transaction

LABELS:
- isFraud: Binary fraud indicator (0=legitimate, 1=fraudulent)
- isMoneyLaundering: Binary AML label (0=normal, 1=suspicious)
- laundering_typology: Specific money laundering pattern type (structuring, layering, integration, normal)
- fraud_probability: Calculated fraud risk score (may be empty for some transactions)

TEMPORAL FEATURES:
- hour: Hour of transaction (0-23)
- day_of_week: Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
- day_of_month: Day of month (1-31)
- month: Month number (1-12)

METADATA:
- metadata: JSON object containing:
  * timestamp: Exact transaction datetime (datetime object)
  * location: Geographic information (city, state, country, postcode)
  * device_info: Device information (type: Mobile/Web/ATM, OS: Android/iOS/Windows/MacOS, IP address)
  * payment_method: Specific payment method (BSB_Account, CardNumber, PayID, etc.)
  * merchant_info: Merchant details (merchant_id, category, risk_level, avg_transaction) or None for P2P transactions
  * risk_indicators: Risk assessment metrics (amount_vs_average, customer_risk_score, category_risk, risk_score, unusual_time, unusual_location)
  * integration_info: Integration laundering details (type, legitimacy_score, detection_risk, location/sector, total_amount, num_sources, average_amount)
  * structuring: Structuring pattern details (sophistication, threshold_proximity, pattern_size)
  * layering: Detailed layering structure with layer information, splits, accounts, and time delays
  * layering_sophistication: Complexity level of layering operations
  * sophistication: Overall transaction sophistication level (low, medium, high)

DATASET STATISTICS:
- Total transactions: 1,090,173 (1M+)
- Legitimate transactions: 1,088,428 (99.84%)
- Money laundering transactions: 1,745 (0.16%)
- Payment types: 7 different methods
- Transaction categories: Multiple categories including financial services
- Time period: 195-day simulation
- Geographic coverage: Australian cities and postcodes

USAGE:
- Anti-money laundering research and algorithm development
- Financial fraud detection benchmarking
- Machine learning model training and validation
- Academic research in financial crime detection
- Educational purposes and student projects
- Commercial Use: For commercial AML system development and testing, please contact s.huda@griffith.edu.au for licensing permissions.

Licensed under Creative Commons Attribution - Non Commercial 4.0 International License (CC BY-NC 4.0). Free to use for non-commercial purposes with proper attribution. 
See LICENSE.txt for full terms.

CITATION:
If you use this dataset, please cite:
Huda, S., Foo, E., Jadidi, Z., Newton, M.A.H., & Sattar, A. (2025). 
AMLNet: A Knowledge-Based Multi-Agent Framework to Generate and Detect 
Realistic Money Laundering Transactions. Expert Systems with Applications.

CONTACT:
s.huda@griffith.edu.au

VERSION: 2.0
DATE: August 2025
