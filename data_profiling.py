#!/usr/bin/env python3
import os
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ====== CONFIG (EDIT IF NEEDED) ======
S3_BUCKET = "fraud-detection-project-csp554v2"
DATA_KEY = "data/creditcard_2023.csv"
S3_PATH = f"s3://{S3_BUCKET}/{DATA_KEY}"

LABEL_COL = "Class"      # 0 = normal, 1 = fraud
AMOUNT_COL = "Amount"    # transaction amount column


def main():
    # 1. Create Spark session
    spark = (
        SparkSession.builder
        .appName("CreditCardDataProfiling")
        .getOrCreate()
    )

    print("=== 1. Loading dataset from S3 ===")
    print(f"Reading: {S3_PATH}")
    df = spark.read.csv(S3_PATH, header=True, inferSchema=True)

    total_rows = df.count()
    cols = df.columns
    dtypes = dict(df.dtypes)

    print(f"Total rows: {total_rows}")
    print(f"Number of columns: {len(cols)}")
    print(f"Columns: {cols}")

    print("\n=== 2. Quick data preview ===")
    df.show(5, truncate=False)
    df.describe().show()

    profile = {}
    profile["basic_statistics"] = {
        "total_records": total_rows,
        "num_features": len(cols),
        "columns": cols,
    }

    # 3. Fraud / class distribution
    if LABEL_COL in cols:
        print(f"\n=== 3. Class distribution for '{LABEL_COL}' ===")
        class_df = df.groupBy(LABEL_COL).count().orderBy(LABEL_COL)
        class_df.show()

        class_counts = {str(r[LABEL_COL]): r["count"] for r in class_df.collect()}
        total_labels = sum(class_counts.values())
        fraud_count = class_counts.get("1", class_counts.get(1, 0))
        fraud_rate = float(fraud_count) / total_labels if total_labels else 0.0

        print(f"Fraud count (label=1): {fraud_count}")
        print(f"Fraud rate: {fraud_rate * 100:.4f}%")

        profile["fraud_statistics"] = {
            "class_counts": class_counts,
            "fraud_transactions": fraud_count,
            "normal_transactions": total_labels - fraud_count,
            "fraud_rate_percentage": fraud_rate * 100.0,
        }
    else:
        print(f"[WARN] Label column '{LABEL_COL}' not found.")
        profile["fraud_statistics"] = {
            "error": f"Label column '{LABEL_COL}' not found"
        }

    # 4. Missing values per column
    print("\n=== 4. Missing values per column ===")
    missing = {}

    for c in cols:
        col_obj = F.col(c)
        if dtypes[c] in ("double", "float", "int", "bigint"):
            m = df.filter(col_obj.isNull() | F.isnan(col_obj)).count()
        else:
            m = df.filter(col_obj.isNull()).count()

        missing[c] = int(m)
        if m > 0:
            print(f"  {c}: {m} missing")

    profile["data_quality"] = {
        "missing_values": missing,
        "has_missing_values": any(v > 0 for v in missing.values())
    }

    # 5. Duplicate rows
    print("\n=== 5. Duplicate rows ===")
    distinct_rows = df.dropDuplicates().count()
    duplicate_count = total_rows - distinct_rows
    print(f"Duplicate rows: {duplicate_count}")

    profile["data_quality"]["duplicate_rows"] = duplicate_count
    profile["data_quality"]["has_duplicates"] = duplicate_count > 0

    # 6. Correlation with label (numeric features)
    if LABEL_COL in cols and dtypes.get(LABEL_COL) in ("double", "float", "int", "bigint"):
        print(f"\n=== 6. Correlation of numeric features with '{LABEL_COL}' ===")
        numeric_cols = [
            c for c, t in df.dtypes
            if t in ("double", "float", "int", "bigint") and c != LABEL_COL
        ]

        correlations = []
        for c in numeric_cols:
            try:
                corr = df.stat.corr(c, LABEL_COL)
                if corr is not None:
                    correlations.append((c, float(corr)))
            except Exception:
                pass

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        profile["correlation_analysis"] = {
            "class_correlations": correlations
        }

        print("Top 10 features most correlated with Class:")
        for c, corr in correlations[:10]:
            print(f"  {c}: {corr:.4f}")
    else:
        print("\n[INFO] Skipping correlation: label not numeric or missing.")
        profile["correlation_analysis"] = {
            "note": "label not numeric or missing"
        }

    # 7. Amount sanity checks
    if AMOUNT_COL in cols:
        print(f"\n=== 7. Amount checks ({AMOUNT_COL}) ===")
        neg_amounts = df.filter(F.col(AMOUNT_COL) < 0).count()
        amount_stats = df.select(
            F.min(AMOUNT_COL).alias("min"),
            F.percentile_approx(AMOUNT_COL, 0.5).alias("median"),
            F.avg(AMOUNT_COL).alias("mean"),
            F.max(AMOUNT_COL).alias("max")
        ).collect()[0]

        print(f"Negative amounts: {neg_amounts}")
        print(f"Min / Median / Mean / Max: "
              f"{amount_stats['min']} / {amount_stats['median']} / "
              f"{amount_stats['mean']} / {amount_stats['max']}")

        profile["amount_checks"] = {
            "negative_amounts": neg_amounts,
            "min": float(amount_stats["min"]),
            "median": float(amount_stats["median"]),
            "mean": float(amount_stats["mean"]),
            "max": float(amount_stats["max"]),
        }

    # 8. Save profile locally and to S3
    print("\n=== 8. Saving profile ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = "/home/hadoop/project"
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, f"profile_metadata_{timestamp}.json")
    with open(local_path, "w") as f:
        json.dump(profile, f, indent=2, default=float)

    print(f"Local profile file: {local_path}")

    s3_profile_key = f"profiling/profile_metadata_{timestamp}.json"
    s3_uri = f"s3://{S3_BUCKET}/{s3_profile_key}"
    # EMR AMI has awscli installed; this uploads the JSON
    os.system(f"aws s3 cp {local_path} {s3_uri}")

    print(f"Uploaded profile to {s3_uri}")
    print("\n=== Profiling complete ===")

    spark.stop()


if __name__ == "__main__":
    main()
