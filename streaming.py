#!/usr/bin/env python3
"""
Streaming inference with the trained GBT model.

- Watches an S3 folder for new CSV files (simulated stream).
- Applies the saved GBT PipelineModel to each micro-batch.
- Writes predictions to S3.

Usage (from ~/project):

spark-submit notebooks/streaming/stream_gbt.py \
  --gbt_model_path s3://fraud-detection-project-csp554v2/models/gbt/model_20251119_031349/
"""

import os
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel


# ===== CONFIG =====
S3_BUCKET = "fraud-detection-project-csp554v2"

# Original full dataset (used only to infer schema)
S3_DATA_PATH = f"s3://{S3_BUCKET}/data/creditcard_2023.csv"

# Streaming folders
S3_STREAM_INPUT = f"s3://{S3_BUCKET}/streaming/input"
S3_STREAM_OUTPUT = f"s3://{S3_BUCKET}/streaming/output/gbt"
S3_CHECKPOINT = f"s3://{S3_BUCKET}/streaming/checkpoints/gbt"

LABEL_COL = "Class"


def create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def infer_schema_from_full_data(spark: SparkSession):
    """
    Read the full CSV once to get a schema for streaming.
    Structured Streaming requires an explicit schema.
    """
    print(f"Inferring schema from {S3_DATA_PATH} ...")
    static_df = spark.read.csv(S3_DATA_PATH, header=True, inferSchema=True)
    print(f"Schema inferred with columns: {static_df.columns}")
    return static_df.schema


def parse_args():
    parser = argparse.ArgumentParser(description="GBT Streaming Inference")

    parser.add_argument(
        "--gbt_model_path",
        required=True,
        help="S3 path to GBT model directory (e.g., s3://.../models/gbt/model_YYYYMMDD_HHMMSS/)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    spark = create_spark_session("GBT_Streaming")

    # 1. Load schema from full dataset
    schema = infer_schema_from_full_data(spark)

    # 2. Load trained GBT pipeline model
    print(f"Loading GBT model from {args.gbt_model_path} ...")
    model = PipelineModel.load(args.gbt_model_path)

    # 3. Define streaming source (S3 folder)
    print(f"Starting stream from input folder: {S3_STREAM_INPUT}")
    stream_df = (
        spark.readStream
        .schema(schema)
        .option("header", True)
        .option("maxFilesPerTrigger", 1)  # process 1 new file per micro-batch
        .csv(S3_STREAM_INPUT)
    )

    # 4. Apply model to streaming data
    #    The pipeline expects features: V1..V28 + Amount; extra columns (id, Class) are allowed.
    preds = model.transform(stream_df)

    # For output, keep some useful columns
    # If any of these don't exist in your data, Spark will complain.
    select_cols = []
    for c in ["id", "Amount", LABEL_COL]:
        if c in preds.columns:
            select_cols.append(c)

    # Always include prediction and probability
    select_cols += ["prediction", "probability"]

    result_df = preds.select(*select_cols)

    # 5. Define streaming sink (write predictions to S3)
    print(f"Writing streaming predictions to: {S3_STREAM_OUTPUT}")
    print(f"Checkpoint location: {S3_CHECKPOINT}")

    query = (
        result_df.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", S3_STREAM_OUTPUT)
        .option("checkpointLocation", S3_CHECKPOINT)
        .start()
    )

    print("Streaming query started. Waiting for data...")
    query.awaitTermination()


if __name__ == "__main__":
    main()
