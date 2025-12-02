#!/usr/bin/env python3
"""
Robustness evaluation for the GBT model.

Scenarios:
  - clean_test: original test set (baseline)
  - noise_10pct: Gaussian noise with 10% of feature std
  - noise_20pct: Gaussian noise with 20% of feature std
  - amount_scaled_1.2: Amount column scaled by 1.2 (distribution shift)

Results are saved as JSON locally and to S3.
"""

import os
import json
import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ===== CONFIG =====
S3_BUCKET = "fraud-detection-project-csp554v2"

S3_TEST_PATH = f"s3://{S3_BUCKET}/processed/test"
S3_ROBUSTNESS_OUTPUT_BASE = f"s3://{S3_BUCKET}/outputs/robustness"

LOCAL_OUTPUT_DIR = "/home/hadoop/project/outputs/robustness"

LABEL_COL = "Class"


def create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_test_data(spark: SparkSession):
    print(f"Loading clean test set from {S3_TEST_PATH} ...")
    df = spark.read.parquet(S3_TEST_PATH)
    print(f"Loaded test set with {df.count():,} rows and columns: {df.columns}")
    return df


def compute_confusion_metrics(pred_df, label_col=LABEL_COL, prediction_col="prediction"):
    """Confusion matrix, precision, recall, F1, accuracy for label==1."""
    tp = pred_df.filter((F.col(prediction_col) == 1) & (F.col(label_col) == 1)).count()
    tn = pred_df.filter((F.col(prediction_col) == 0) & (F.col(label_col) == 0)).count()
    fp = pred_df.filter((F.col(prediction_col) == 1) & (F.col(label_col) == 0)).count()
    fn = pred_df.filter((F.col(prediction_col) == 0) & (F.col(label_col) == 1)).count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_model_on_df(model, df, scenario_name: str):
    """Run model on df and compute metrics."""
    print(f"\n=== Evaluating scenario: {scenario_name} ===")
    pred = model.transform(df)

    evaluator_roc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    auc_roc = evaluator_roc.evaluate(pred)
    auc_pr = evaluator_pr.evaluate(pred)

    conf_metrics = compute_confusion_metrics(pred)

    metrics = {
        "scenario": scenario_name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        **conf_metrics,
    }

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def add_gaussian_noise(df, std_fraction: float, exclude_cols):
    """
    Add Gaussian noise N(0, (std_fraction * std_col)^2) to numeric columns,
    excluding label and any columns in exclude_cols.
    """
    print(f"Applying Gaussian noise with std_fraction={std_fraction} ...")

    noisy_df = df
    numeric_cols = [
        c for c, t in df.dtypes
        if t in ("double", "float", "int", "bigint") and c not in exclude_cols
    ]

    # Pre-compute std dev for each numeric column
    stats = df.select([
        F.stddev(F.col(c)).alias(c) for c in numeric_cols
    ]).collect()[0].asDict()

    for c in numeric_cols:
        std_val = stats.get(c, None)
        if std_val is None or std_val == 0:
            continue

        noise_col = F.randn(seed=42) * float(std_val) * std_fraction
        noisy_df = noisy_df.withColumn(c, F.col(c) + noise_col)

    return noisy_df


def scale_amount(df, factor: float, amount_col: str = "Amount"):
    """Scale the Amount column by a constant factor."""
    if amount_col not in df.columns:
        print(f"[WARN] Amount column '{amount_col}' not found. Skipping scaling.")
        return df
    print(f"Scaling '{amount_col}' by factor {factor} ...")
    return df.withColumn(amount_col, F.col(amount_col) * F.lit(factor))


def parse_args():
    parser = argparse.ArgumentParser(description="GBT robustness evaluation")

    parser.add_argument(
        "--gbt_model_path",
        required=True,
        help="S3 path to GBT model directory (e.g., s3://.../models/gbt/model_YYYYMMDD_HHMMSS/)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    spark = create_spark_session("GBT_Robustness")
    test_df = load_test_data(spark)

    print(f"Loading GBT model from {args.gbt_model_path} ...")
    model = PipelineModel.load(args.gbt_model_path)

    # Common columns to never perturb
    exclude_cols = {LABEL_COL}

    results = []

    # 1) Clean baseline
    results.append(evaluate_model_on_df(model, test_df, "clean_test"))

    # 2) Noise 10%
    noisy_10 = add_gaussian_noise(test_df, std_fraction=0.10, exclude_cols=exclude_cols)
    results.append(evaluate_model_on_df(model, noisy_10, "noise_10pct"))

    # 3) Noise 20%
    noisy_20 = add_gaussian_noise(test_df, std_fraction=0.20, exclude_cols=exclude_cols)
    results.append(evaluate_model_on_df(model, noisy_20, "noise_20pct"))

    # 4) Amount scaled 1.2
    scaled_amount = scale_amount(test_df, factor=1.2, amount_col="Amount")
    results.append(evaluate_model_on_df(model, scaled_amount, "amount_scaled_1.2"))

    # === Save robustness results ===
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    local_path = os.path.join(LOCAL_OUTPUT_DIR, f"robustness_gbt_{timestamp}.json")
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nSaved robustness results locally to {local_path}")

    s3_path = f"{S3_ROBUSTNESS_OUTPUT_BASE}/robustness_gbt_{timestamp}.json"
    os.system(f"aws s3 cp {local_path} {s3_path}")
    print(f"Uploaded robustness results to {s3_path}")

    spark.stop()
    print("\n=== Robustness evaluation complete. ===")


if __name__ == "__main__":
    main()
