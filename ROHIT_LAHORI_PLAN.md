# Rohit Lahori - Technical Lead / ML Pipeline Architect

## CSP 554 Big Data Technologies - Fraud Detection Project

**Role:** Technical Lead / ML Pipeline Architect  
**Timeline:** November 18 - December 10, 2025  
**Primary Focus:** EMR infrastructure, Spark MLlib pipelines, model training, streaming integration

---

## Table of Contents

1. [Role Overview](#role-overview)
2. [Phase-by-Phase Execution Plan](#phase-by-phase-execution-plan)
3. [File Structure](#file-structure)
4. [Setup & Configuration](#setup--configuration)
5. [Code Implementations](#code-implementations)
6. [Integration Points](#integration-points)
7. [Quality Gates](#quality-gates)
8. [Daily Task Breakdown](#daily-task-breakdown)
9. [Checkpoint Questions](#checkpoint-questions)
10. [Pro Tips](#pro-tips)

---

## Role Overview

### Primary Responsibilities

1. **Infrastructure Setup**

   - AWS EMR cluster creation and configuration
   - EMR Studio workspace setup
   - S3 bucket structure and IAM policies
   - GitHub repository initialization

2. **ML Pipeline Development**

   - Feature engineering pipelines
   - Model training with Spark MLlib (LR, RF, GBT)
   - Hyperparameter space definition
   - Model persistence and versioning

3. **Performance Optimization**

   - Caching strategies
   - Partitioning optimization
   - Resource monitoring and auto-scaling
   - Error handling for distributed computing

4. **Streaming Integration**

   - Structured Streaming architecture
   - Batch-to-streaming model deployment
   - Performance tuning (batch intervals, watermarking)
   - Model serving in streaming context

5. **Deployment & Demo**
   - Code cleanup and documentation
   - Demo script for end-to-end pipeline
   - Performance benchmark reports
   - Architecture documentation

### Success Metrics

- ✅ EMR cluster operational with 3 nodes
- ✅ All three models trained and saved to S3
- ✅ Pipeline runtime < 5 minutes on full dataset
- ✅ Streaming integration functional
- ✅ Demo script runs end-to-end successfully

### Integration with Team

- **Receives from Hussain:** Data profiling reports, quality validation
- **Provides to Ansh:** Trained models, feature-engineered data
- **Provides to Hussain:** Pipeline components for testing
- **Coordinates:** Infrastructure setup, Git repository management

---

## Phase-by-Phase Execution Plan

### Phase 1: Infrastructure & Base Pipeline (Nov 18-22)

**Goal:** Set up AWS EMR, create base MLlib pipeline

#### Day 1 (Nov 18): AWS Setup

**Tasks:**

1. Create S3 bucket
2. Upload dataset (Phase 0 from Master Plan)
3. Set up IAM roles and policies
4. Create EMR cluster

#### Day 2-3 (Nov 19-20): EMR Studio & Project Structure

**Tasks:**

1. Create EMR Studio workspace
2. Initialize Git repository
3. Create project structure
4. Set up Spark configuration

#### Day 4-5 (Nov 21-22): Base Pipeline Implementation

**Tasks:**

1. Data loading pipeline
2. Feature engineering
3. Baseline model training (Logistic Regression)

**Deliverable:** Working EMR cluster with base pipeline

---

### Phase 2: Complete ML Pipeline (Nov 23-27)

**Goal:** Implement all three models with optimization

#### Day 1-2 (Nov 23-24): Feature Engineering & Scaling

**Tasks:**

1. Complete feature engineering pipeline
2. Implement StandardScaler
3. VectorAssembler configuration

#### Day 3-4 (Nov 25-26): Model Training

**Tasks:**

1. Train Random Forest
2. Train GBT Classifier
3. Model persistence to S3

#### Day 5 (Nov 27): Optimization & Testing

**Tasks:**

1. Implement caching strategies
2. Performance benchmarking
3. Integration with Ansh's evaluation framework

**Deliverable:** All three models trained and optimized

---

### Phase 3: Advanced Features (Nov 28-Dec 3)

**Goal:** Adversarial training, ensemble, optimization

#### Day 1-2 (Nov 28-29): Adversarial Training Integration

**Tasks:**

1. Integrate adversarial examples into training
2. Adversarial training loop
3. Model versioning for adversarial models

#### Day 3-4 (Nov 30-Dec 1): Ensemble & Optimization

**Tasks:**

1. Implement ensemble voting classifier
2. Performance optimization (partitioning, caching)
3. Resource monitoring setup

#### Day 5-6 (Dec 2-3): Error Handling & Resilience

**Tasks:**

1. Error handling for node failures
2. Data skew handling
3. Auto-scaling configuration

**Deliverable:** Robust, optimized pipeline with adversarial training

---

### Phase 4: Streaming Integration (Dec 4-6)

**Goal:** Integrate models into streaming context

#### Day 1-2 (Dec 4-5): Streaming Architecture

**Tasks:**

1. Structured Streaming setup
2. Model loading in streaming context
3. Checkpointing configuration

#### Day 3 (Dec 6): Performance Tuning

**Tasks:**

1. Batch interval optimization
2. Watermarking configuration
3. Batch vs. streaming comparison

**Deliverable:** Streaming pipeline operational

---

### Phase 5: Deployment & Documentation (Dec 7-10)

**Goal:** Finalize code, create demo, document architecture

#### Day 1-2 (Dec 7-8): Code Cleanup

**Tasks:**

1. Code refactoring
2. Documentation (docstrings, comments)
3. README creation

#### Day 3-4 (Dec 9-10): Demo & Documentation

**Tasks:**

1. Demo script development
2. Performance benchmark report
3. Architecture diagram updates

**Deliverable:** Production-ready codebase with demo

---

## File Structure

```
src/rohit/
├── __init__.py
├── infrastructure/
│   ├── __init__.py
│   ├── emr_setup.sh                 # EMR cluster creation script
│   ├── iam_policies.json            # IAM policy templates
│   └── spark_config.py              # Spark configuration
├── pipelines/
│   ├── __init__.py
│   ├── base_pipeline.py             # Base pipeline class
│   ├── logistic_regression.py       # LR pipeline
│   ├── random_forest.py             # RF pipeline
│   └── gbt_classifier.py            # GBT pipeline
├── optimization/
│   ├── __init__.py
│   ├── caching_strategies.py        # Caching utilities
│   ├── partitioning.py              # Partitioning strategies
│   └── resource_monitor.py          # Resource monitoring
├── streaming/
│   ├── __init__.py
│   ├── streaming_integration.py     # Streaming setup
│   └── model_serving.py             # Model serving in streaming
└── deployment/
    ├── __init__.py
    ├── model_versioning.py          # Model versioning system
    └── demo_script.py               # End-to-end demo
```

---

## Setup & Configuration

### AWS CLI Setup

```bash
# Install AWS CLI (if not installed)
# Windows: Download from https://aws.amazon.com/cli/
# Linux/Mac: pip install awscli

# Configure AWS CLI
aws configure
# Enter:
# - AWS Access Key ID: [Your access key]
# - AWS Secret Access Key: [Your secret key]
# - Default region: us-east-1
# - Default output format: json

# Verify configuration
aws sts get-caller-identity
```

### S3 Bucket Creation

```bash
# Create S3 bucket (replace YOUR_BUCKET_NAME with unique name)
export BUCKET_NAME="csp554-fraud-detection-$(date +%s)"
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Set bucket policy (see IAM section below)
aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://bucket-policy.json

# Verify bucket
aws s3 ls s3://$BUCKET_NAME/
```

### IAM Roles & Policies

**File:** `src/rohit/infrastructure/iam_policies.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::csp554-fraud-detection-*/*",
        "arn:aws:s3:::csp554-fraud-detection-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

**Create IAM Role for EMR:**

```bash
# Create role (via AWS Console or CLI)
aws iam create-role \
    --role-name EMR_DefaultRole \
    --assume-role-policy-document file://emr-trust-policy.json

# Attach policy
aws iam put-role-policy \
    --role-name EMR_DefaultRole \
    --policy-name FraudDetectionPolicy \
    --policy-document file://src/rohit/infrastructure/iam_policies.json
```

### EMR Cluster Creation

**File:** `src/rohit/infrastructure/emr_setup.sh`

```bash
#!/bin/bash

# EMR Cluster Setup Script
# Usage: ./emr_setup.sh <cluster-name> <bucket-name>

CLUSTER_NAME=${1:-"fraud-detection-cluster"}
BUCKET_NAME=${2:-"csp554-fraud-detection-default"}

echo "Creating EMR cluster: $CLUSTER_NAME"

# Create cluster
aws emr create-cluster \
    --name "$CLUSTER_NAME" \
    --release-label emr-6.15.0 \
    --instance-type m5.xlarge \
    --instance-count 3 \
    --applications Name=Spark Name=Hadoop Name=JupyterEnterpriseGateway \
    --ec2-attributes KeyName=your-key-pair,InstanceProfile=EMR_DefaultRole \
    --log-uri s3://$BUCKET_NAME/emr-logs/ \
    --configurations file://emr-config.json \
    --auto-terminate \
    --region us-east-1

# Get cluster ID
CLUSTER_ID=$(aws emr list-clusters \
    --cluster-states WAITING RUNNING \
    --query "Clusters[?Name=='$CLUSTER_NAME'].Id" \
    --output text)

echo "Cluster ID: $CLUSTER_ID"
echo "Waiting for cluster to be ready..."

# Wait for cluster to be ready
aws emr wait cluster-running --cluster-id $CLUSTER_ID

echo "Cluster is ready!"
echo "Master node DNS:"
aws emr describe-cluster --cluster-id $CLUSTER_ID \
    --query "Cluster.MasterPublicDnsName" \
    --output text
```

**EMR Configuration File:** `emr-config.json`

```json
[
  {
    "Classification": "spark-defaults",
    "Properties": {
      "spark.sql.adaptive.enabled": "true",
      "spark.sql.adaptive.coalescePartitions.enabled": "true",
      "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
      "spark.executor.memory": "4g",
      "spark.executor.cores": "2",
      "spark.driver.memory": "2g",
      "spark.sql.shuffle.partitions": "200"
    }
  },
  {
    "Classification": "spark-env",
    "Properties": {},
    "Configurations": [
      {
        "Classification": "export",
        "Properties": {
          "PYSPARK_PYTHON": "/usr/bin/python3"
        }
      }
    ]
  }
]
```

**Run setup:**

```bash
chmod +x src/rohit/infrastructure/emr_setup.sh
./src/rohit/infrastructure/emr_setup.sh fraud-detection-cluster $BUCKET_NAME
```

### EMR Studio Setup

```bash
# Create EMR Studio (via AWS Console recommended)
# Or use CLI:
aws emr create-studio \
    --name fraud-detection-workspace \
    --auth-mode SSO \
    --default-s3-location s3://$BUCKET_NAME/emr-studio/ \
    --region us-east-1

# Get Studio ID
STUDIO_ID=$(aws emr list-studios --query "Studios[?Name=='fraud-detection-workspace'].StudioId" --output text)

# Create workspace
aws emr create-workspace \
    --studio-id $STUDIO_ID \
    --name fraud-detection-workspace
```

### GitHub Repository Setup

```bash
# Initialize Git repository
git init
git branch -M main

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Spark
*.log
checkpoint/
spark-warehouse/

# AWS
.aws/
*.pem

# IDE
.idea/
.vscode/
*.swp

# Data
data/raw/*.csv
data/processed/

# Models (large files)
models/*.pkl
*.model

# Environment
.env
EOF

# Initial commit
git add .
git commit -m "Initial commit: Project structure"

# Create remote repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/BigData_FinalProject.git
git push -u origin main

# Set up branch protection (via GitHub UI)
# Settings > Branches > Add rule for 'main'
# Require: Pull request reviews, status checks
```

### Spark Configuration

**File:** `src/rohit/infrastructure/spark_config.py`

```python
"""
Spark configuration for EMR cluster.
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from src.common.config import config

def create_optimized_spark_session(app_name: str = "FraudDetectionPipeline"):
    """
    Create optimized Spark session for EMR.

    Args:
        app_name: Application name

    Returns:
        Configured SparkSession
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("yarn") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .config("spark.sql.files.maxPartitionBytes", "134217728") \
        .config("spark.sql.files.openCostInBytes", "4194304") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark

# Usage
if __name__ == "__main__":
    spark = create_optimized_spark_session()
    print(f"Spark version: {spark.version}")
    print(f"Master: {spark.sparkContext.master}")
    spark.stop()
```

---

## Code Implementations

### Phase 1: Base Pipeline

**File:** `src/rohit/pipelines/base_pipeline.py`

```python
"""
Base pipeline class for fraud detection models.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import functions as F
from typing import List, Dict
from src.common.config import config
from src.common.schema_validator import get_credit_card_schema

class BaseFraudDetectionPipeline:
    """Base class for fraud detection pipelines."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        self.label_col = "Class"

    def load_data(self, s3_path: str = None) -> DataFrame:
        """
        Load credit card fraud dataset from S3.

        Args:
            s3_path: S3 path to dataset (uses config if None)

        Returns:
            DataFrame with loaded data
        """
        if s3_path is None:
            s3_path = config.S3_RAW_DATA_PATH

        schema = get_credit_card_schema()
        df = self.spark.read.schema(schema).csv(s3_path, header=True)

        # Rename Class to label for MLlib
        df = df.withColumnRenamed("Class", "label")

        print(f"Loaded {df.count():,} records from {s3_path}")
        return df

    def create_feature_pipeline(self) -> Pipeline:
        """
        Create feature engineering pipeline.

        Returns:
            Pipeline with VectorAssembler and StandardScaler
        """
        # VectorAssembler: Combine features into single vector
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features"
        )

        # StandardScaler: Normalize features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )

        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler])

        return pipeline

    def train_test_split(
        self,
        df: DataFrame,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> tuple:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            train_ratio: Proportion for training
            seed: Random seed

        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

        print(f"Train size: {train_df.count():,}")
        print(f"Test size: {test_df.count():,}")

        return train_df, test_df

    def save_model(
        self,
        model: PipelineModel,
        model_name: str,
        version: str = "v1"
    ):
        """
        Save trained model to S3.

        Args:
            model: Trained PipelineModel
            model_name: Name of model (e.g., "logistic_regression")
            version: Model version
        """
        model_path = f"{config.S3_MODELS_PATH}/{model_name}/{version}/model/"

        # Save model
        model.write().overwrite().save(model_path)

        # Save metadata
        import json
        import boto3
        from datetime import datetime

        metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "feature_cols": self.feature_cols,
            "label_col": self.label_col
        }

        s3_client = boto3.client('s3')
        metadata_key = f"{config.S3_MODELS_PATH}/{model_name}/{version}/metadata.json"
        bucket, key = metadata_key.replace("s3://", "").split("/", 1)

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metadata, indent=2),
            ContentType="application/json"
        )

        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_key}")

    @staticmethod
    def load_model(model_path: str) -> PipelineModel:
        """
        Load trained model from S3.

        Args:
            model_path: S3 path to model

        Returns:
            Loaded PipelineModel
        """
        model = PipelineModel.load(model_path)
        print(f"Model loaded from {model_path}")
        return model

# Usage example
if __name__ == "__main__":
    from src.rohit.infrastructure.spark_config import create_optimized_spark_session

    spark = create_optimized_spark_session("BasePipelineTest")

    # Create pipeline
    pipeline = BaseFraudDetectionPipeline(spark)

    # Load data
    df = pipeline.load_data()

    # Split data
    train_df, test_df = pipeline.train_test_split(df)

    # Save splits
    train_df.write.mode("overwrite").parquet(f"{config.S3_PROCESSED_PATH}/train/")
    test_df.write.mode("overwrite").parquet(f"{config.S3_PROCESSED_PATH}/test/")

    spark.stop()
```

**File:** `src/rohit/pipelines/logistic_regression.py`

```python
"""
Logistic Regression pipeline for fraud detection.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline
from src.common.config import config

class LogisticRegressionPipeline(BaseFraudDetectionPipeline):
    """Logistic Regression pipeline implementation."""

    def __init__(self, spark: SparkSession):
        super().__init__(spark)
        self.model_name = "logistic_regression"

    def create_model_pipeline(self) -> Pipeline:
        """
        Create complete pipeline with Logistic Regression.

        Returns:
            Pipeline with feature engineering and classifier
        """
        # Feature pipeline
        feature_pipeline = self.create_feature_pipeline()

        # Logistic Regression
        lr = LogisticRegression(
            featuresCol="scaled_features",
            labelCol="label",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0
        )

        # Complete pipeline
        pipeline = Pipeline(stages=feature_pipeline.getStages() + [lr])

        return pipeline

    def train(
        self,
        train_df: DataFrame,
        save_model: bool = True,
        version: str = "v1"
    ) -> PipelineModel:
        """
        Train Logistic Regression model.

        Args:
            train_df: Training DataFrame
            save_model: Whether to save model to S3
            version: Model version

        Returns:
            Trained PipelineModel
        """
        print("Training Logistic Regression model...")

        # Create pipeline
        pipeline = self.create_model_pipeline()

        # Fit model
        model = pipeline.fit(train_df)

        # Save if requested
        if save_model:
            self.save_model(model, self.model_name, version)

        print("Logistic Regression training completed!")
        return model

    def evaluate(
        self,
        model: PipelineModel,
        test_df: DataFrame
    ) -> dict:
        """
        Evaluate model on test data.

        Args:
            model: Trained PipelineModel
            test_df: Test DataFrame

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = model.transform(test_df)

        # Calculate AUROC
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auroc = evaluator.evaluate(predictions)

        # Calculate AUPRC
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )
        auprc = evaluator_pr.evaluate(predictions)

        metrics = {
            "auroc": auroc,
            "auprc": auprc
        }

        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        return metrics

# Usage
if __name__ == "__main__":
    from src.rohit.infrastructure.spark_config import create_optimized_spark_session

    spark = create_optimized_spark_session("LogisticRegressionTraining")

    # Create pipeline
    lr_pipeline = LogisticRegressionPipeline(spark)

    # Load data
    df = lr_pipeline.load_data()

    # Split data
    train_df, test_df = lr_pipeline.train_test_split(df)

    # Train model
    model = lr_pipeline.train(train_df, save_model=True)

    # Evaluate
    metrics = lr_pipeline.evaluate(model, test_df)

    spark.stop()
```

**File:** `src/rohit/pipelines/random_forest.py`

```python
"""
Random Forest pipeline for fraud detection.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline
from src.common.config import config

class RandomForestPipeline(BaseFraudDetectionPipeline):
    """Random Forest pipeline implementation."""

    def __init__(self, spark: SparkSession):
        super().__init__(spark)
        self.model_name = "random_forest"

    def create_model_pipeline(self) -> Pipeline:
        """Create complete pipeline with Random Forest."""
        feature_pipeline = self.create_feature_pipeline()

        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol="label",
            numTrees=100,
            maxDepth=10,
            impurity="gini",
            seed=42
        )

        pipeline = Pipeline(stages=feature_pipeline.getStages() + [rf])
        return pipeline

    def train(
        self,
        train_df: DataFrame,
        save_model: bool = True,
        version: str = "v1"
    ) -> PipelineModel:
        """Train Random Forest model."""
        print("Training Random Forest model...")

        pipeline = self.create_model_pipeline()
        model = pipeline.fit(train_df)

        if save_model:
            self.save_model(model, self.model_name, version)

        print("Random Forest training completed!")
        return model

    def get_feature_importance(self, model: PipelineModel) -> dict:
        """Extract feature importance from Random Forest model."""
        rf_model = model.stages[-1]  # Get RF model from pipeline

        importances = rf_model.featureImportances.toArray()
        feature_importance = {
            self.feature_cols[i]: float(importances[i])
            for i in range(len(self.feature_cols))
        }

        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

# Similar implementation for GBT Classifier...
# (See full code in repository)
```

### Phase 2: Optimization

**File:** `src/rohit/optimization/caching_strategies.py`

```python
"""
Caching strategies for Spark DataFrames.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark import StorageLevel

class CachingManager:
    """Manage DataFrame caching for performance optimization."""

    @staticmethod
    def cache_dataframe(df: DataFrame, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK):
        """
        Cache DataFrame with specified storage level.

        Args:
            df: DataFrame to cache
            storage_level: Storage level (MEMORY_ONLY, MEMORY_AND_DISK, etc.)

        Returns:
            Cached DataFrame
        """
        return df.persist(storage_level)

    @staticmethod
    def unpersist_dataframe(df: DataFrame):
        """Unpersist cached DataFrame."""
        df.unpersist()

    @staticmethod
    def cache_if_large(df: DataFrame, threshold_rows: int = 100000):
        """
        Cache DataFrame only if it's large enough.

        Args:
            df: DataFrame to potentially cache
            threshold_rows: Minimum rows to trigger caching

        Returns:
            DataFrame (cached if large)
        """
        row_count = df.count()
        if row_count > threshold_rows:
            print(f"Caching DataFrame with {row_count:,} rows")
            return CachingManager.cache_dataframe(df)
        else:
            print(f"DataFrame has {row_count:,} rows, skipping cache")
            return df

# Usage
if __name__ == "__main__":
    from src.rohit.infrastructure.spark_config import create_optimized_spark_session

    spark = create_optimized_spark_session()

    # Load data
    df = spark.read.csv("s3://bucket/raw-data/creditcard.csv", header=True)

    # Cache if large
    df = CachingManager.cache_if_large(df)

    # Use DataFrame multiple times
    # ... operations ...

    # Unpersist when done
    CachingManager.unpersist_dataframe(df)

    spark.stop()
```

**File:** `src/rohit/optimization/partitioning.py`

```python
"""
Partitioning strategies for Spark optimization.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

class PartitioningStrategy:
    """Partitioning utilities for data optimization."""

    @staticmethod
    def repartition_by_label(df: DataFrame, label_col: str = "label", num_partitions: int = None):
        """
        Repartition DataFrame by label for balanced distribution.

        Args:
            df: Input DataFrame
            label_col: Name of label column
            num_partitions: Number of partitions (uses default if None)

        Returns:
            Repartitioned DataFrame
        """
        if num_partitions is None:
            num_partitions = df.rdd.getNumPartitions()

        return df.repartition(num_partitions, label_col)

    @staticmethod
    def coalesce_partitions(df: DataFrame, num_partitions: int):
        """
        Coalesce partitions to reduce overhead.

        Args:
            df: Input DataFrame
            num_partitions: Target number of partitions

        Returns:
            Coalesced DataFrame
        """
        return df.coalesce(num_partitions)

    @staticmethod
    def optimize_partitions_for_skew(df: DataFrame, key_col: str):
        """
        Optimize partitions for skewed data.

        Args:
            df: Input DataFrame
            key_col: Column with potential skew

        Returns:
            Optimized DataFrame
        """
        # Add salt to handle skew
        df_with_salt = df.withColumn("salt", F.rand())
        return df_with_salt.repartition("salt", key_col)
```

### Phase 3: Adversarial Training

**File:** `src/rohit/pipelines/adversarial_training.py`

```python
"""
Adversarial training integration.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline
from src.ansh.adversarial.fgsm_attack import FGSMAttack

class AdversarialTrainingPipeline(BaseFraudDetectionPipeline):
    """Pipeline with adversarial training support."""

    def train_with_adversarial_examples(
        self,
        train_df: DataFrame,
        base_model: PipelineModel,
        epsilon: float = 0.1,
        adversarial_ratio: float = 0.1
    ) -> PipelineModel:
        """
        Train model with adversarial examples.

        Args:
            train_df: Clean training data
            base_model: Pre-trained base model
            epsilon: Perturbation magnitude
            adversarial_ratio: Proportion of adversarial examples

        Returns:
            Adversarially trained model
        """
        # Generate adversarial examples
        attack = FGSMAttack(self.spark, base_model, epsilon=epsilon)
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Sample subset for adversarial examples
        n_adversarial = int(train_df.count() * adversarial_ratio)
        sample_df = train_df.sample(False, adversarial_ratio, seed=42).limit(n_adversarial)

        adversarial_df = attack.generate_adversarial_examples(sample_df, feature_cols)

        # Combine clean and adversarial data
        combined_df = train_df.union(adversarial_df)

        # Retrain model
        pipeline = self.create_model_pipeline()
        adversarially_trained_model = pipeline.fit(combined_df)

        print(f"Trained with {train_df.count():,} clean + {adversarial_df.count():,} adversarial examples")

        return adversarially_trained_model
```

### Phase 4: Streaming Integration

**File:** `src/rohit/streaming/streaming_integration.py`

```python
"""
Structured Streaming integration for fraud detection.
"""
from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQuery
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from src.common.config import config

class StreamingFraudDetection:
    """Fraud detection in streaming context."""

    def __init__(self, spark: SparkSession, model: PipelineModel):
        self.spark = spark
        self.model = model

    def create_streaming_pipeline(
        self,
        input_path: str,
        checkpoint_location: str,
        output_mode: str = "append"
    ) -> StreamingQuery:
        """
        Create streaming pipeline for fraud detection.

        Args:
            input_path: S3 path to streaming data
            checkpoint_location: S3 checkpoint path
            output_mode: Output mode (append, complete, update)

        Returns:
            StreamingQuery
        """
        # Read streaming data
        streaming_df = self.spark.readStream \
            .schema(get_credit_card_schema()) \
            .option("maxFilesPerTrigger", 10) \
            .csv(input_path)

        # Rename Class to label
        streaming_df = streaming_df.withColumnRenamed("Class", "label")

        # Apply model
        predictions = self.model.transform(streaming_df)

        # Add fraud probability and prediction
        predictions = predictions.withColumn(
            "fraud_probability",
            F.col("probability")[1]
        ).withColumn(
            "is_fraud",
            F.when(F.col("prediction") == 1, True).otherwise(False)
        )

        # Select relevant columns
        output_df = predictions.select(
            "Time", "Amount", "fraud_probability", "is_fraud", "prediction"
        )

        # Write stream
        query = output_df.writeStream \
            .format("parquet") \
            .outputMode(output_mode) \
            .option("checkpointLocation", checkpoint_location) \
            .option("path", f"{config.S3_OUTPUTS_PATH}/streaming_predictions/") \
            .trigger(processingTime="10 seconds") \
            .start()

        return query

# Usage
if __name__ == "__main__":
    from src.rohit.infrastructure.spark_config import create_optimized_spark_session
    from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline

    spark = create_optimized_spark_session("StreamingFraudDetection")

    # Load model
    model = BaseFraudDetectionPipeline.load_model(
        f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/"
    )

    # Create streaming pipeline
    streaming = StreamingFraudDetection(spark, model)
    query = streaming.create_streaming_pipeline(
        input_path=f"{config.S3_PROCESSED_PATH}/streaming/",
        checkpoint_location=f"{config.S3_BUCKET}/checkpoints/streaming/"
    )

    # Run for specified time
    query.awaitTermination(timeout=300)  # 5 minutes

    spark.stop()
```

### Phase 5: Demo Script

**File:** `src/rohit/deployment/demo_script.py`

```python
"""
End-to-end demo script for fraud detection pipeline.
"""
from pyspark.sql import SparkSession
from src.rohit.infrastructure.spark_config import create_optimized_spark_session
from src.rohit.pipelines.logistic_regression import LogisticRegressionPipeline
from src.rohit.pipelines.random_forest import RandomForestPipeline
from src.rohit.pipelines.gbt_classifier import GBTPipeline
from src.ansh.evaluation.metrics import FraudDetectionMetrics
from src.common.config import config
import time

def run_demo():
    """Run complete fraud detection demo."""
    print("=" * 60)
    print("Fraud Detection Pipeline Demo")
    print("=" * 60)

    # Initialize Spark
    spark = create_optimized_spark_session("FraudDetectionDemo")

    try:
        # 1. Load data
        print("\n[1/5] Loading data...")
        start_time = time.time()
        base_pipeline = LogisticRegressionPipeline(spark)
        df = base_pipeline.load_data()
        print(f"✓ Loaded {df.count():,} records in {time.time() - start_time:.2f}s")

        # 2. Split data
        print("\n[2/5] Splitting data...")
        train_df, test_df = base_pipeline.train_test_split(df)

        # 3. Train models
        print("\n[3/5] Training models...")
        models = {}

        # Logistic Regression
        print("  Training Logistic Regression...")
        lr_pipeline = LogisticRegressionPipeline(spark)
        models["lr"] = lr_pipeline.train(train_df, save_model=True)

        # Random Forest
        print("  Training Random Forest...")
        rf_pipeline = RandomForestPipeline(spark)
        models["rf"] = rf_pipeline.train(train_df, save_model=True)

        # 4. Evaluate models
        print("\n[4/5] Evaluating models...")
        metrics_calc = FraudDetectionMetrics(spark)

        for model_name, model in models.items():
            predictions = model.transform(test_df)
            metrics = metrics_calc.calculate_all_metrics(predictions)

            print(f"\n{model_name.upper()} Results:")
            print(f"  AUROC: {metrics['auroc']:.4f}")
            print(f"  AUPRC: {metrics['auprc']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")

        # 5. Summary
        print("\n[5/5] Demo completed successfully!")
        print(f"Total runtime: {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    run_demo()
```

---

## Integration Points

### Receiving Data from Hussain

```python
# Load profiling metadata
import json
import boto3

s3_client = boto3.client('s3')
response = s3_client.get_object(
    Bucket=config.S3_BUCKET,
    Key="profiling/metadata_latest.json"
)
profiling_metadata = json.loads(response['Body'].read().decode('utf-8'))

# Use for feature selection or validation
quality_checks = profiling_metadata.get('quality_checks', {})
if not quality_checks.get('data_quality_passed', False):
    raise ValueError("Data quality checks failed!")
```

### Providing Models to Ansh

```python
# Models are saved to S3 with versioning
# Ansh can load them using:
from pyspark.ml import PipelineModel

model = PipelineModel.load(f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/")
```

### Providing Components to Hussain

```python
# Export pipeline functions for testing
from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline

# Hussain can import and test
```

---

## Quality Gates

### Cluster Health Checks

```bash
# Check cluster status
aws emr describe-cluster --cluster-id <cluster-id> --query "Cluster.Status.State"

# Check node health
aws emr list-instances --cluster-id <cluster-id> --query "Instances[].Status.State"
```

### Pipeline Validation

```python
# Validate pipeline with sample data
def validate_pipeline(pipeline, sample_df):
    try:
        model = pipeline.fit(sample_df)
        predictions = model.transform(sample_df)
        assert predictions.count() == sample_df.count()
        print("✓ Pipeline validation passed")
        return True
    except Exception as e:
        print(f"✗ Pipeline validation failed: {e}")
        return False
```

### Performance Benchmarks

- Pipeline runtime < 5 minutes on full dataset
- Memory usage < 80% of allocated
- CPU utilization > 50% during training

---

## Daily Task Breakdown

### Phase 1 (Nov 18-22)

| Day    | Task                 | Hours | Deliverable          |
| ------ | -------------------- | ----- | -------------------- |
| Nov 18 | AWS setup, S3, IAM   | 6     | Infrastructure ready |
| Nov 19 | EMR cluster creation | 4     | Cluster operational  |
| Nov 20 | EMR Studio setup     | 4     | Workspace ready      |
| Nov 21 | Base pipeline        | 6     | base_pipeline.py     |
| Nov 22 | LR model training    | 4     | First model trained  |

### Phase 2 (Nov 23-27)

| Day    | Task                | Hours | Deliverable           |
| ------ | ------------------- | ----- | --------------------- |
| Nov 23 | Feature engineering | 4     | Complete pipeline     |
| Nov 24 | RF model            | 4     | RF trained            |
| Nov 25 | GBT model           | 4     | GBT trained           |
| Nov 26 | Optimization        | 6     | Caching, partitioning |
| Nov 27 | Integration testing | 4     | All models working    |

### Phase 3 (Nov 28-Dec 3)

| Day     | Task                 | Hours | Deliverable          |
| ------- | -------------------- | ----- | -------------------- |
| Nov 28  | Adversarial training | 6     | Adversarial pipeline |
| Nov 29  | Ensemble model       | 4     | Voting classifier    |
| Nov 30  | Performance tuning   | 6     | Optimized pipeline   |
| Dec 1   | Error handling       | 4     | Resilient pipeline   |
| Dec 2-3 | Testing              | 4     | All tests pass       |

### Phase 4 (Dec 4-6)

| Day   | Task               | Hours | Deliverable           |
| ----- | ------------------ | ----- | --------------------- |
| Dec 4 | Streaming setup    | 6     | Streaming pipeline    |
| Dec 5 | Model serving      | 4     | Streaming operational |
| Dec 6 | Performance tuning | 4     | Optimized streaming   |

### Phase 5 (Dec 7-10)

| Day    | Task          | Hours | Deliverable        |
| ------ | ------------- | ----- | ------------------ |
| Dec 7  | Code cleanup  | 6     | Clean codebase     |
| Dec 8  | Documentation | 4     | README, docstrings |
| Dec 9  | Demo script   | 4     | Working demo       |
| Dec 10 | Final testing | 4     | Production ready   |

---

## Checkpoint Questions

### After Phase 1

- [ ] Can you create and access EMR cluster?
- [ ] Is data loading from S3 working?
- [ ] Can you train a baseline model?

### After Phase 2

- [ ] Are all three models trained?
- [ ] Are models saved to S3 correctly?
- [ ] Is pipeline runtime < 5 minutes?

### After Phase 3

- [ ] Is adversarial training integrated?
- [ ] Is ensemble model working?
- [ ] Are error handling mechanisms in place?

### After Phase 4

- [ ] Is streaming pipeline operational?
- [ ] Are checkpoints working?
- [ ] Is model serving in streaming context?

### After Phase 5

- [ ] Does demo script run end-to-end?
- [ ] Is documentation complete?
- [ ] Are all integration points working?

---

## Pro Tips

1. **Use Spot Instances:** Save 70% on EMR costs with Spot instances
2. **Monitor Resources:** Set up CloudWatch alarms for cluster health
3. **Version Models:** Always version models for reproducibility
4. **Cache Strategically:** Cache DataFrames used multiple times
5. **Partition Wisely:** Use appropriate partitioning for your data size
6. **Error Handling:** Implement retry logic for transient failures
7. **Logging:** Use structured logging for debugging
8. **Testing:** Test pipelines with small datasets first
9. **Documentation:** Document all configuration parameters
10. **Cost Control:** Set up budget alerts to avoid surprises

---

**Document Version:** 1.0  
**Last Updated:** November 18, 2025  
**Owner:** Rohit Lahori
