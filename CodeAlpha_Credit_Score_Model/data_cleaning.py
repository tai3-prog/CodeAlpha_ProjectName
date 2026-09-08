"""
CodeAlpha ML Internship - Credit Scoring Model
Data Cleaning & Preprocessing Script

This script ONLY cleans and preprocesses the data.
No model training, no scaling, no SMOTE, no target encoding.
"""

import pandas as pd
import numpy as np
import re

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# =========================================================
# 1. LOAD DATA
# =========================================================
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

df = pd.read_csv("train.csv")

print(f"Shape: {df.shape}")

# =========================================================
# 2. INITIAL INSPECTION
# =========================================================
print("\n" + "=" * 70)
print("STEP 2: INITIAL INSPECTION")
print("=" * 70)

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values (top 15) ---")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print(f"\n--- Duplicated rows: {df.duplicated().sum()} ---")

print("\n--- Target distribution (Credit_Score) ---")
print(df["Credit_Score"].value_counts(dropna=False))

# =========================================================
# 3. REMOVE IDENTIFIER / PII COLUMNS
# =========================================================
print("\n" + "=" * 70)
print("STEP 3: REMOVING IDENTIFIER / PII COLUMNS")
print("=" * 70)

# ID                -> a row-level unique identifier, carries no predictive signal
# Customer_ID       -> identifies a specific person; using it risks the model
#                      memorizing individuals rather than learning general patterns,
#                      and it would leak identity across the Month-based repeated rows
# Name              -> personally identifying, no predictive value, privacy concern
# SSN               -> personally identifying (sensitive PII), must never be used as a feature
id_cols_to_drop = ["ID", "Customer_ID", "Name", "SSN"]
df = df.drop(columns=[c for c in id_cols_to_drop if c in df.columns])

print(f"Dropped columns: {id_cols_to_drop}")
print(f"Shape after drop: {df.shape}")

# =========================================================
# 4. HELPER FUNCTIONS FOR CLEANING "DIRTY" NUMERIC STRINGS
# =========================================================
print("\n" + "=" * 70)
print("STEP 4: CLEANING NUMERIC COLUMNS STORED AS OBJECT")
print("=" * 70)


def clean_numeric_string(series: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series of strings that should represent numbers but
    contain junk characters commonly found in this dataset, e.g.:
        "23_"          -> "23"
        "_______"      -> NaN (nothing left after cleaning)
        "1000.25_"     -> "1000.25"
        "-500"         -> "-500"   (kept as-is; validity checked later)
        "NM", "nan"    -> NaN
    Strategy:
        1. Convert to string.
        2. Strip whitespace.
        3. Remove trailing/leading underscores (a recurring artifact in this
           dataset, e.g. Age values like "28_").
        4. Replace known "missing" placeholder tokens with NaN.
        5. Let pd.to_numeric(..., errors="coerce") turn anything still
           invalid into NaN rather than crashing or guessing.
    """
    s = series.astype(str).str.strip()

    # Remove stray underscores anywhere in the string (e.g. "28_", "_28", "2_8")
    s = s.str.replace("_", "", regex=False)

    # Normalize known placeholder / junk tokens to NaN
    junk_tokens = ["", "nan", "NaN", "NM", "None", "!@9#%8", "#F%$D@*&8"]
    s = s.replace(junk_tokens, np.nan)

    return pd.to_numeric(s, errors="coerce")


numeric_object_cols = [
    "Age",
    "Annual_Income",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

for col in numeric_object_cols:
    if col in df.columns:
        before_dtype = df[col].dtype
        df[col] = clean_numeric_string(df[col])
        print(f"  {col}: {before_dtype} -> {df[col].dtype} "
              f"({df[col].isnull().sum()} NaN after cleaning)")

# =========================================================
# 5. HANDLE OBVIOUSLY INVALID NUMERIC VALUES
# =========================================================
print("\n" + "=" * 70)
print("STEP 5: HANDLING INVALID / IMPOSSIBLE NUMERIC VALUES")
print("=" * 70)

# --- Age ---
# Real-world age cannot be negative or absurdly large (e.g. 8000).
# Rather than deleting rows, we treat impossible ages as missing (NaN) so
# they get median-imputed later. A reasonable plausible human range is used.
if "Age" in df.columns:
    invalid_age = (df["Age"] < 0) | (df["Age"] > 100)
    print(f"  Age: {invalid_age.sum()} invalid values found (set to NaN)")
    df.loc[invalid_age, "Age"] = np.nan

# --- Annual_Income ---
# Income cannot be negative. Extremely large values are kept (income can
# genuinely be high) but negatives are treated as data entry errors.
if "Annual_Income" in df.columns:
    invalid_income = df["Annual_Income"] < 0
    print(f"  Annual_Income: {invalid_income.sum()} negative values found (set to NaN)")
    df.loc[invalid_income, "Annual_Income"] = np.nan

# --- Num_Bank_Accounts ---
# Cannot be negative; an unreasonably high count (e.g. > 20) is almost
# certainly a data error rather than a real customer, so treat as missing.
if "Num_Bank_Accounts" in df.columns:
    invalid_bank = (df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20)
    print(f"  Num_Bank_Accounts: {invalid_bank.sum()} invalid values found (set to NaN)")
    df.loc[invalid_bank, "Num_Bank_Accounts"] = np.nan

# --- Num_Credit_Card ---
if "Num_Credit_Card" in df.columns:
    invalid_cc = (df["Num_Credit_Card"] < 0) | (df["Num_Credit_Card"] > 20)
    print(f"  Num_Credit_Card: {invalid_cc.sum()} invalid values found (set to NaN)")
    df.loc[invalid_cc, "Num_Credit_Card"] = np.nan

# --- Interest_Rate ---
# Interest rate as a percentage should not be negative or absurdly high
# (e.g. > 100% is treated as a data entry problem for this dataset).
if "Interest_Rate" in df.columns:
    invalid_rate = (df["Interest_Rate"] < 0) | (df["Interest_Rate"] > 100)
    print(f"  Interest_Rate: {invalid_rate.sum()} invalid values found (set to NaN)")
    df.loc[invalid_rate, "Interest_Rate"] = np.nan

# --- Num_of_Loan ---
# Cannot be negative; unreasonably large loan counts are treated as errors.
if "Num_of_Loan" in df.columns:
    invalid_loan = (df["Num_of_Loan"] < 0) | (df["Num_of_Loan"] > 20)
    print(f"  Num_of_Loan: {invalid_loan.sum()} invalid values found (set to NaN)")
    df.loc[invalid_loan, "Num_of_Loan"] = np.nan

# --- Num_of_Delayed_Payment ---
if "Num_of_Delayed_Payment" in df.columns:
    invalid_delay = (df["Num_of_Delayed_Payment"] < 0) | (df["Num_of_Delayed_Payment"] > 30)
    print(f"  Num_of_Delayed_Payment: {invalid_delay.sum()} invalid values found (set to NaN)")
    df.loc[invalid_delay, "Num_of_Delayed_Payment"] = np.nan

# --- Num_Credit_Inquiries ---
if "Num_Credit_Inquiries" in df.columns:
    invalid_inq = df["Num_Credit_Inquiries"] < 0
    print(f"  Num_Credit_Inquiries: {invalid_inq.sum()} negative values found (set to NaN)")
    df.loc[invalid_inq, "Num_Credit_Inquiries"] = np.nan

# --- Outstanding_Debt ---
# Debt cannot be negative.
if "Outstanding_Debt" in df.columns:
    invalid_debt = df["Outstanding_Debt"] < 0
    print(f"  Outstanding_Debt: {invalid_debt.sum()} negative values found (set to NaN)")
    df.loc[invalid_debt, "Outstanding_Debt"] = np.nan

# --- Credit_Utilization_Ratio ---
# This is typically a percentage (0-100). Negative or > 100 is invalid.
if "Credit_Utilization_Ratio" in df.columns:
    invalid_util = (df["Credit_Utilization_Ratio"] < 0) | (df["Credit_Utilization_Ratio"] > 100)
    print(f"  Credit_Utilization_Ratio: {invalid_util.sum()} invalid values found (set to NaN)")
    df.loc[invalid_util, "Credit_Utilization_Ratio"] = np.nan

# --- Total_EMI_per_month / Amount_invested_monthly / Monthly_Balance ---
# These are monetary values and cannot be negative.
for col in ["Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]:
    if col in df.columns:
        invalid_money = df[col] < 0
        print(f"  {col}: {invalid_money.sum()} negative values found (set to NaN)")
        df.loc[invalid_money, col] = np.nan

# =========================================================
# 6. Credit_History_Age -> Credit_History_Months
# =========================================================
print("\n" + "=" * 70)
print("STEP 6: CONVERTING Credit_History_Age TO Credit_History_Months")
print("=" * 70)

# The raw column stores strings like "22 Years and 1 Months", which are not
# directly usable by a model. We parse the years and months out with a
# regex and combine them into a single numeric feature (total months),
# which is far more useful for a model than a text string.


def parse_credit_history(value):
    if pd.isnull(value):
        return np.nan
    value = str(value)
    match = re.search(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", value, re.IGNORECASE)
    if match:
        years, months = int(match.group(1)), int(match.group(2))
        return years * 12 + months
    return np.nan


if "Credit_History_Age" in df.columns:
    df["Credit_History_Months"] = df["Credit_History_Age"].apply(parse_credit_history)
    print(f"  Parsed {df['Credit_History_Months'].notnull().sum()} valid values")
    print(f"  {df['Credit_History_Months'].isnull().sum()} missing after parsing")

    # Decision: drop the original text column. All of its information is now
    # captured numerically in Credit_History_Months, and keeping the raw
    # string column would be redundant and unusable by ML models directly.
    df = df.drop(columns=["Credit_History_Age"])
    print("  Original 'Credit_History_Age' dropped (info now in Credit_History_Months)")

# =========================================================
# 7. Type_of_Loan FEATURE ENGINEERING
# =========================================================
print("\n" + "=" * 70)
print("STEP 7: FEATURE ENGINEERING FOR Type_of_Loan")
print("=" * 70)

# Type_of_Loan contains comma-separated lists of loan types per customer,
# e.g. "Auto Loan, Credit-Builder Loan, Personal Loan". One-hot encoding
# this directly would create an unreasonable number of sparse dummy columns.
# Instead we engineer two simple, informative features:
#   - Num_Loan_Types: how many distinct loan types the customer holds
#   - Has_Loan: whether the customer holds any loan at all
if "Type_of_Loan" in df.columns:
    def count_loan_types(value):
        if pd.isnull(value) or str(value).strip().lower() in ["", "nan", "not specified"]:
            return 0
        return len([x for x in str(value).split(",") if x.strip() != ""])

    df["Num_Loan_Types"] = df["Type_of_Loan"].apply(count_loan_types)
    df["Has_Loan"] = (df["Num_Loan_Types"] > 0).astype(int)

    print(f"  Created 'Num_Loan_Types' (range {df['Num_Loan_Types'].min()}-{df['Num_Loan_Types'].max()})")
    print(f"  Created 'Has_Loan' (value counts): {df['Has_Loan'].value_counts().to_dict()}")

    # Decision: drop the original free-text column now that its useful
    # signal has been distilled into the two numeric features above.
    df = df.drop(columns=["Type_of_Loan"])
    print("  Original 'Type_of_Loan' dropped (info now in Num_Loan_Types / Has_Loan)")

# =========================================================
# 8. CLEAN SUSPICIOUS / JUNK VALUES IN CATEGORICAL COLUMNS
# =========================================================
print("\n" + "=" * 70)
print("STEP 8: CLEANING SUSPICIOUS VALUES IN CATEGORICAL COLUMNS")
print("=" * 70)

categorical_cols = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Month",
]

# Known junk/placeholder tokens observed in this dataset for categorical
# columns. These get converted to NaN so they can be imputed consistently
# rather than being treated as a legitimate category.
categorical_junk_tokens = ["_______", "NM", "nan", "NaN", "", "!@9#%8", "#F%$D@*&8", "Not Specified"]

for col in categorical_cols:
    if col in df.columns:
        n_before = df[col].isin(categorical_junk_tokens).sum()
        df[col] = df[col].replace(categorical_junk_tokens, np.nan)
        print(f"  {col}: {n_before} junk values replaced with NaN")

# =========================================================
# 9. CHECK MISSING VALUES
# =========================================================
print("\n" + "=" * 70)
print("STEP 9: CHECKING MISSING VALUES")
print("=" * 70)

# IMPORTANT:
# We do NOT impute missing values here.
#
# Imputation will be done later inside a Scikit-learn Pipeline
# AFTER train/test split.
#
# This prevents data leakage because statistics such as the median
# will be learned only from the training data.

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if len(missing_values) > 0:
    print("\nMissing values that will be handled later:")
    print(missing_values)
else:
    print("\nNo missing values found.")

    
# =========================================================
# 10. REMOVE DUPLICATE ROWS
# =========================================================
print("\n" + "=" * 70)
print("STEP 10: REMOVING DUPLICATE ROWS")
print("=" * 70)

n_dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"  Removed {n_dupes} exact duplicate rows")

# =========================================================
# 11. FINAL SANITY CHECK - NO TARGET LEAKAGE
# =========================================================
# Credit_Score is the target column and must remain untouched as the label,
# never used to derive/compute any other feature above. Confirmed: none of
# the feature engineering steps above referenced Credit_Score.
assert "Credit_Score" in df.columns, "Target column missing!"

# =========================================================
# 12. FINAL SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("STEP 12: FINAL SUMMARY")
print("=" * 70)

print(f"\nFinal shape: {df.shape}")

print("\n--- Final data types ---")
print(df.dtypes)

print("\n--- Remaining missing values ---")
remaining_na = df.isnull().sum()
print(remaining_na[remaining_na > 0] if remaining_na.sum() > 0 else "None")

print(f"\n--- Duplicate count: {df.duplicated().sum()} ---")

print("\n--- Target distribution (Credit_Score) ---")
print(df["Credit_Score"].value_counts())

print("\n--- First 5 rows ---")
print(df.head())

# =========================================================
# 13. SAVE CLEANED DATASET
# =========================================================
df.to_csv("credit_cleaned.csv", index=False)
print("\nSaved cleaned dataset to 'credit_cleaned.csv'")