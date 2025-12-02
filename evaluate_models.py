#!/usr/bin/env python3
"""
Evaluate Logistic Regression, Random Forest, and GBT models
on the shared test set and compare their performance.

Usage (from ~/project):

spark-submit notebooks/evaluation/evaluate_models.py \
  --lr_model_path  s3://fraud-detection-project-csp554v2/models/lr/model_YYYYMMDD_HHMMSS \
  --rf_model_path  s3://fraud-detection-project-csp554v2/models/rf/model_YYYYMMDD_HHMMSS \
  --gbt_model_path s3://fraud-detection-project-csp554v2/models/gbt/model_YYYYMMDD_HHMMSS
"""

import os
import json
import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ===== GLOBAL CONFIG =====
S3_BUCKET = "fraud-detection-project-csp554v2"

S3_TEST_PATH = f"s3://{S3_BUCKET}/processed/test"
S3_EVAL_OUTPUT_BASE = f"s3://{S3_BUCKET}/outputs/evaluation"

LOCAL_OUTPUT_DIR = "/home/hadoop/project/outputs"

LABEL_COL = "Class"


def create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_test_data(spark: SparkSession):
    print(f"Loading test set from {S3_TEST_PATH} ...")
    df = spark.read.parquet(S3_TEST_PATH)
    print(f"Loaded test set with {df.count():,} rows and columns: {df.columns}")
    return df


def compute_confusion_metrics(pred_df, label_col=LABEL_COL, prediction_col="prediction"):
    """
    Compute confusion matrix components + precision, recall, F1, accuracy
    for positive class (label==1).
    """
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


def evaluate_model(model_name: str, model_path: str, test_df):
    """
    Load a PipelineModel from S3, run predictions on test_df, and compute metrics.
    """
    print(f"\n=== Evaluating {model_name} ===")
    print(f"Loading model from {model_path} ...")
    model = PipelineModel.load(model_path)

    pred = model.transform(test_df)

    # AUC-ROC and AUC-PR
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
        "model_name": model_name,
        "model_path": model_path,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        **conf_metrics,
    }

    print(f"Metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def save_metrics_all(models_metrics):
    """
    Save per-model metrics JSONs and a combined comparison JSON,
    both locally and to S3.
    """
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save individual metrics
    json_paths_local = []

    for m in models_metrics:
        model_name = m["model_name"]
        filename = f"metrics_{model_name}_{timestamp}.json"
        local_path = os.path.join(LOCAL_OUTPUT_DIR, filename)
        with open(local_path, "w") as f:
            json.dump(m, f, indent=2, default=float)

        json_paths_local.append((model_name, local_path))
        print(f"Saved {model_name} metrics locally to {local_path}")

    # 2. Save combined comparison JSON
    combined_filename = f"metrics_comparison_{timestamp}.json"
    combined_local = os.path.join(LOCAL_OUTPUT_DIR, combined_filename)
    with open(combined_local, "w") as f:
        json.dump(models_metrics, f, indent=2, default=float)

    print(f"Saved combined metrics locally to {combined_local}")

    # 3. Copy all metrics to S3
    s3_base = S3_EVAL_OUTPUT_BASE

    # per-model files
    for model_name, local_path in json_paths_local:
        s3_key = f"{s3_base}/metrics_{model_name}_{timestamp}.json"
        os.system(f"aws s3 cp {local_path} {s3_key}")
        print(f"Uploaded {model_name} metrics to {s3_key}")

    # combined file
    combined_s3_key = f"{s3_base}/{combined_filename}"
    os.system(f"aws s3 cp {combined_local} {combined_s3_key}")
    print(f"Uploaded comparison file to {combined_s3_key}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LR, RF, and GBT models")

    parser.add_argument(
        "--lr_model_path",
        required=True,
        help="S3 path to Logistic Regression model directory"
    )
    parser.add_argument(
        "--rf_model_path",
        required=True,
        help="S3 path to Random Forest model directory"
    )
    parser.add_argument(
        "--gbt_model_path",
        required=True,
        help="S3 path to Gradient-Boosted Trees model directory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    spark = create_spark_session("Evaluate_Models")
    test_df = load_test_data(spark)

    all_metrics = []

    # Evaluate each model
    all_metrics.append(
        evaluate_model("logistic_regression", args.lr_model_path, test_df)
    )
    all_metrics.append(
        evaluate_model("random_forest", args.rf_model_path, test_df)
    )
    all_metrics.append(
        evaluate_model("gbt", args.gbt_model_path, test_df)
    )

    # Save and upload metrics
    save_metrics_all(all_metrics)

    spark.stop()
    print("\n=== Evaluation complete. Metrics saved to outputs/ and S3. ===")


if __name__ == "__main__":
    main()
