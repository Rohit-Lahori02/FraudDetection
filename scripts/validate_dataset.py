"""
Dataset validation script for credit card fraud detection.
Validates dataset schema, statistics, and quality.
"""
import pandas as pd
import os
from pathlib import Path

def validate_dataset(csv_path=None):
    """
    Validate credit card fraud dataset locally.
    
    Args:
        csv_path: Path to CSV file. Defaults to creditcard_2023.csv/creditcard_2023.csv
    """
    if csv_path is None:
        # Default path as specified in the plan
        csv_path = os.path.join('creditcard_2023.csv', 'creditcard_2023.csv')
    
    # Ensure path exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset file not found at {csv_path}")
        print(f"Current working directory: {os.getcwd()}")
        return None
    
    print(f"Reading dataset from: {csv_path}")
    print("=" * 60)
    
    # Read sample for quick validation
    df_sample = pd.read_csv(csv_path, nrows=1000)
    
    print(f"Columns ({len(df_sample.columns)}): {df_sample.columns.tolist()}")
    print(f"Sample shape: {df_sample.shape}")
    print(f"Missing values in sample: {df_sample.isnull().sum().sum()}")
    
    if 'Class' in df_sample.columns:
        fraud_rate = df_sample['Class'].mean()
        print(f"Fraud rate in sample: {fraud_rate:.4%}")
        fraud_count = df_sample['Class'].sum()
        print(f"Fraud transactions in sample: {fraud_count} out of {len(df_sample)}")
    
    # Read full dataset for complete statistics
    print("\nReading full dataset for complete statistics...")
    df_full = pd.read_csv(csv_path)
    
    print(f"\nFull dataset shape: {df_full.shape}")
    print(f"Total missing values: {df_full.isnull().sum().sum()}")
    
    if 'Class' in df_full.columns:
        fraud_rate_full = df_full['Class'].mean()
        fraud_count_full = df_full['Class'].sum()
        print(f"Total fraud transactions: {fraud_count_full:,} out of {len(df_full):,}")
        print(f"Overall fraud rate: {fraud_rate_full:.4%}")
    
    # Column data types
    print("\nColumn data types:")
    for col in df_full.columns:
        print(f"  {col}: {df_full[col].dtype}")
    
    # Missing values per column
    missing_per_col = df_full.isnull().sum()
    if missing_per_col.sum() > 0:
        print("\nMissing values per column:")
        for col, count in missing_per_col.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(df_full)*100:.2f}%)")
    else:
        print("\nNo missing values found in any column.")
    
    # Basic statistics for numeric columns
    print("\nBasic statistics for Amount column:")
    if 'Amount' in df_full.columns:
        print(df_full['Amount'].describe())
    
    return df_full

if __name__ == "__main__":
    df = validate_dataset()
    if df is not None:
        print("\n" + "=" * 60)
        print("Dataset validation completed successfully!")

