#!/usr/bin/env python3
"""
Logistic Regression training script.
Feature engineering + model training only.
Evaluation will be done in a separate script.
"""

import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression


# ===== CONFIG =====
S3_BUCKET = "fraud-detection-project-csp554v2"
S3_DATA_PATH = f"s3://{S3_BUCKET}/data/creditcard_2023.csv"

S3_PROCESSED_BASE = f"s3://{S3_BUCKET}/processed"
S3_TRAIN_PATH = f"{S3_PROCESSED_BASE}/train"
S3_TEST_PATH = f"{S3_PROCESSED_BASE}/test"

S3_MODEL_BASE = f"s3://{S3_BUCKET}/models/lr"

LABEL_COL = "Class"


def create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_and_prepare_data(spark: SparkSession):
    print(f"Loading dataset from {S3_DATA_PATH} ...")
    df = spark.read.csv(S3_DATA_PATH, header=True, inferSchema=True)
    print(f"Loaded {df.count():,} rows with columns: {df.columns}")

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop("id")

    # Cast label to int
    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("int"))

    # Feature columns
    feature_cols = [c for c in df.columns if c != LABEL_COL]

    return df, feature_cols


def create_lr_pipeline(feature_cols):
    """
    Feature engineering + Logistic Regression model.
    - VectorAssembler -> 'features_raw'
    - StandardScaler -> 'features'
    - LogisticRegression uses 'features'
    """
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    lr = LogisticRegression(
        labelCol=LABEL_COL,
        featuresCol="features",
        maxIter=20,
        regParam=0.01
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])
    return pipeline


def main():
    spark = create_spark_session("LR_Training")

    # 1. Load and prepare data
    df, feature_cols = load_and_prepare_data(spark)

    # 2. Train / test split (shared for all models)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train size: {train_df.count():,} | Test size: {test_df.count():,}")

    # 3. Persist split to S3 for evaluation scripts
    print(f"Writing train/test splits to {S3_PROCESSED_BASE} ...")
    train_df.write.mode("overwrite").parquet(S3_TRAIN_PATH)
    test_df.write.mode("overwrite").parquet(S3_TEST_PATH)

    # 4. Create pipeline and train model (on train set only)
    pipeline = create_lr_pipeline(feature_cols)

    print("Training Logistic Regression model ...")
    model = pipeline.fit(train_df)
    print("Training complete.")

    # 5. Save model to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{S3_MODEL_BASE}/model_{timestamp}"
    model.write().overwrite().save(model_path)

    print(f"Model saved to: {model_path}")
    print("No evaluation done here – use separate evaluation script.")

    spark.stop()


if __name__ == "__main__":
    main()
