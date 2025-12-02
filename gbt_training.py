#!/usr/bin/env python3
"""
Gradient-Boosted Trees (GBT) training script.
Feature engineering + model training only.
Evaluation will be done in a separate script.
"""

from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier


S3_BUCKET = "fraud-detection-project-csp554v2"
S3_DATA_PATH = f"s3://{S3_BUCKET}/data/creditcard_2023.csv"

S3_PROCESSED_BASE = f"s3://{S3_BUCKET}/processed"
S3_TRAIN_PATH = f"{S3_PROCESSED_BASE}/train"
S3_TEST_PATH = f"{S3_PROCESSED_BASE}/test"

S3_MODEL_BASE = f"s3://{S3_BUCKET}/models/gbt"

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

    if "id" in df.columns:
        df = df.drop("id")

    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("int"))

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    return df, feature_cols


def create_gbt_pipeline(feature_cols):
    """
    Feature engineering + GBT model.
    - VectorAssembler -> 'features'
    """
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    gbt = GBTClassifier(
        labelCol=LABEL_COL,
        featuresCol="features",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, gbt])
    return pipeline


def main():
    spark = create_spark_session("GBT_Training")

    # 1. Load and prepare data
    df, feature_cols = load_and_prepare_data(spark)

    # 2. Train / test split (same seed; same processed paths)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train size: {train_df.count():,} | Test size: {test_df.count():,}")

    train_df.write.mode("overwrite").parquet(S3_TRAIN_PATH)
    test_df.write.mode("overwrite").parquet(S3_TEST_PATH)

    # 3. Create pipeline and train model
    pipeline = create_gbt_pipeline(feature_cols)

    print("Training GBT model ...")
    model = pipeline.fit(train_df)
    print("Training complete.")

    # 4. Save model to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{S3_MODEL_BASE}/model_{timestamp}"
    model.write().overwrite().save(model_path)

    print(f"Model saved to: {model_path}")
    print("No evaluation done here – use separate evaluation script.")

    spark.stop()


if __name__ == "__main__":
    main()
