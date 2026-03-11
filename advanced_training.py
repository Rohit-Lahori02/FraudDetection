#!/usr/bin/env python3
"""
Advanced Training Script with Phase 1 Enhancements:
1. Temporal Feature Engineering (Velocity, Time-since-last)
2. SMOTE Integration for Class Balancing
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from imblearn.over_sampling import SMOTE

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier

# ===== CONFIG =====
S3_BUCKET = "fraud-detection-project-csp554v2"
S3_DATA_PATH = f"s3://{S3_BUCKET}/data/creditcard_2023.csv"
LABEL_COL = "Class"
ID_COL = "id"
AMOUNT_COL = "Amount"

def create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )

def add_temporal_features(df):
    """
    Simulate temporal features. 
    Since the dataset is anonymized (V1-V28), we assume 'id' represents sequence.
    In a real scenario, we'd use a 'Timestamp' column.
    """
    print("Adding temporal features (simulated via sequence)...")
    
    # Define a window ordered by id (proxy for time)
    window_spec = Window.orderBy(ID_COL)
    
    # 1. Time-since-last (simulated as id difference)
    df = df.withColumn("prev_id", F.lag(ID_COL).over(window_spec))
    df = df.withColumn("id_delta", F.when(F.col("prev_id").isNull(), 0).otherwise(F.col(ID_COL) - F.col("prev_id")))
    
    # 2. Transaction Velocity (Rolling average of Amount over last 10 transactions)
    window_velocity = Window.orderBy(ID_COL).rowsBetween(-10, 0)
    df = df.withColumn("amount_velocity", F.avg(AMOUNT_COL).over(window_velocity))
    
    # 3. Cumulative Amount
    window_cum = Window.orderBy(ID_COL).rowsBetween(Window.unboundedPreceding, 0)
    df = df.withColumn("cum_amount", F.sum(AMOUNT_COL).over(window_cum))
    
    # Drop intermediate columns
    df = df.drop("prev_id")
    
    return df

def apply_smote(df, feature_cols):
    """
    Apply SMOTE to balance the dataset.
    Note: SMOTE is typically applied to Pandas/NumPy, so we convert a sample or use a distributed approach.
    For this implementation, we'll demonstrate the logic via conversion for the training set.
    """
    print("Applying SMOTE for class balancing...")
    
    # Convert to Pandas for SMOTE (in production, use spark-smote or similar distributed libs)
    # We'll take a significant sample to keep it manageable in the sandbox
    pdf = df.select(feature_cols + [LABEL_COL]).toPandas()
    
    X = pdf[feature_cols]
    y = pdf[LABEL_COL]
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Convert back to Spark DataFrame
    resampled_pdf = pd.concat([pd.DataFrame(X_res, columns=feature_cols), pd.Series(y_res, name=LABEL_COL)], axis=1)
    
    # Initialize Spark session again to ensure it's available for conversion
    spark = SparkSession.builder.getOrCreate()
    resampled_df = spark.createDataFrame(resampled_pdf)
    
    return resampled_df

def main():
    spark = create_spark_session("Advanced_Fraud_Training")
    
    # 1. Load Data
    print(f"Loading data from {S3_DATA_PATH}...")
    df = spark.read.csv(S3_DATA_PATH, header=True, inferSchema=True)
    
    # 2. Temporal Feature Engineering
    df = add_temporal_features(df)
    
    # 3. Prepare Features
    # Exclude id and Class from features
    feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]
    print(f"Feature columns: {feature_cols}")
    
    # 4. Train/Test Split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # 5. Apply SMOTE to Training Data
    # Note: In a real large-scale EMR job, you'd use a distributed SMOTE implementation.
    # Here we demonstrate the integration logic.
    balanced_train_df = apply_smote(train_df, feature_cols)
    
    print(f"Original train size: {train_df.count()}")
    print(f"Balanced train size: {balanced_train_df.count()}")
    
    # 6. Build Pipeline
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    
    gbt = GBTClassifier(
        labelCol=LABEL_COL,
        featuresCol="scaled_features",
        maxIter=20
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, gbt])
    
    # 7. Train Model
    print("Training advanced GBT model with temporal features and SMOTE...")
    model = pipeline.fit(balanced_train_df)
    
    # 8. Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"s3://{S3_BUCKET}/models/advanced_gbt/model_{timestamp}"
    print(f"Model training complete. (Simulated save to {model_path})")
    
    spark.stop()

if __name__ == "__main__":
    main()
