# Credit Card Fraud Dataset Schema Documentation

## Dataset Overview

- **Location**: `creditcard_2023.csv/creditcard_2023.csv`
- **Total Records**: 568,630
- **Total Columns**: 31
- **Missing Values**: 0 (no missing values in any column)

## Schema Definition

| Column Name | Data Type | Description                             |
| ----------- | --------- | --------------------------------------- |
| id          | int64     | Transaction identifier                  |
| V1          | float64   | Principal component feature 1           |
| V2          | float64   | Principal component feature 2           |
| V3          | float64   | Principal component feature 3           |
| V4          | float64   | Principal component feature 4           |
| V5          | float64   | Principal component feature 5           |
| V6          | float64   | Principal component feature 6           |
| V7          | float64   | Principal component feature 7           |
| V8          | float64   | Principal component feature 8           |
| V9          | float64   | Principal component feature 9           |
| V10         | float64   | Principal component feature 10          |
| V11         | float64   | Principal component feature 11          |
| V12         | float64   | Principal component feature 12          |
| V13         | float64   | Principal component feature 13          |
| V14         | float64   | Principal component feature 14          |
| V15         | float64   | Principal component feature 15          |
| V16         | float64   | Principal component feature 16          |
| V17         | float64   | Principal component feature 17          |
| V18         | float64   | Principal component feature 18          |
| V19         | float64   | Principal component feature 19          |
| V20         | float64   | Principal component feature 20          |
| V21         | float64   | Principal component feature 21          |
| V22         | float64   | Principal component feature 22          |
| V23         | float64   | Principal component feature 23          |
| V24         | float64   | Principal component feature 24          |
| V25         | float64   | Principal component feature 25          |
| V26         | float64   | Principal component feature 26          |
| V27         | float64   | Principal component feature 27          |
| V28         | float64   | Principal component feature 28          |
| Amount      | float64   | Transaction amount                      |
| Class       | int64     | Target variable (0 = Normal, 1 = Fraud) |

## Dataset Statistics

### Class Distribution

- **Total Transactions**: 568,630
- **Fraud Transactions**: 284,315 (50.00%)
- **Normal Transactions**: 284,315 (50.00%)
- **Fraud Rate**: 50.00%

### Amount Column Statistics

- **Mean**: 12,041.96
- **Standard Deviation**: 6,919.64
- **Minimum**: 50.01
- **25th Percentile**: 6,054.89
- **Median**: 12,030.15
- **75th Percentile**: 18,036.33
- **Maximum**: 24,039.93

## Data Quality

- **No missing values** in any column
- All columns have expected data types
- Dataset is balanced with 50% fraud and 50% normal transactions

## Notes

- The dataset uses principal component analysis (PCA) transformations for features V1 through V28
- The original features have been transformed for privacy reasons
- The 'id' column serves as a unique identifier for each transaction
- The 'Class' column is the binary target variable for fraud detection (0 = Normal, 1 = Fraud)
