# Master Integration Plan

## CSP 554 Big Data Technologies - Fraud Detection Project

**Team Members:** Ansh Kaushik, Rohit Lahori, Hussain Bin Yousuf  
**Timeline:** November 18 - December 10, 2025  
**Platform:** AWS EMR with Apache Spark MLlib  
**Dataset:** Kaggle Credit Card Fraud Detection (~284K transactions, 0.17% fraud rate)

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Phase 0: Dataset Preparation](#phase-0-dataset-preparation)
3. [Integration Points](#integration-points)
4. [Git Branching Strategy](#git-branching-strategy)
5. [S3 Bucket Structure](#s3-bucket-structure)
6. [EMR Studio Workspace Organization](#emr-studio-workspace-organization)
7. [Dependency Matrix](#dependency-matrix)
8. [Cost Management](#cost-management)
9. [Shared Utilities](#shared-utilities)
10. [Conflict Resolution Protocols](#conflict-resolution-protocols)
11. [Daily Standup Agenda](#daily-standup-agenda)
12. [Code Review Checklist](#code-review-checklist)
13. [Emergency Scenarios & Rollback Procedures](#emergency-scenarios--rollback-procedures)
14. [Deliverable Checklist](#deliverable-checklist)
15. [Final Integration & Testing Protocol](#final-integration--testing-protocol)

---

## Project Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS EMR Cluster                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              EMR Studio Workspace                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │   Rohit's    │  │   Ansh's     │  │  Hussain's   │  │  │
│  │  │   Pipeline   │  │  Evaluation  │  │  Profiling   │  │  │
│  │  │   Notebooks  │  │  Notebooks   │  │  Notebooks   │  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │  │
│  └─────────┼──────────────────┼──────────────────┼─────────┘  │
│            │                  │                  │             │
│  ┌─────────▼──────────────────▼──────────────────▼─────────┐  │
│  │         Shared Spark Session (Common Utilities)          │  │
│  └───────────────────────────┬─────────────────────────────┘  │
└───────────────────────────────┼─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   S3 Data Lake        │
                    │  ┌─────────────────┐ │
                    │  │ raw-data/       │ │
                    │  │ processed/      │ │
                    │  │ models/         │ │
                    │  │ checkpoints/    │ │
                    │  │ outputs/        │ │
                    │  └─────────────────┘ │
                    └──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   GitHub Repository   │
                    │  ┌─────────────────┐ │
                    │  │ main            │ │
                    │  │ feature/*       │ │
                    │  │ src/common/     │ │
                    │  │ src/rohit/      │ │
                    │  │ src/ansh/       │ │
                    │  │ src/hussain/    │ │
                    │  └─────────────────┘ │
                    └──────────────────────┘
```

### Component Flow

```
Dataset (Kaggle)
    ↓
S3 Raw Data (s3://bucket/raw-data/)
    ↓
Hussain: Data Profiling → Quality Report
    ↓
Rohit: Feature Engineering → Model Training
    ↓
Ansh: Model Evaluation → Metrics & Visualizations
    ↓
Hussain: Testing & Validation
    ↓
Rohit: Model Deployment (Streaming)
    ↓
Ansh: Streaming Evaluation
    ↓
Final Deliverables
```

---

## Phase 0: Dataset Preparation

**Timeline:** November 18, 2025 (Day 1)  
**Owner:** All team members (coordinated by Rohit)

### Step 1: Kaggle API Setup

```bash
# Install Kaggle CLI
pip install kaggle

# Create Kaggle API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create API token (download kaggle.json)
# 3. Place in ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<username>\.kaggle\kaggle.json (Windows)

# Set permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json

# Verify setup
kaggle datasets list
```

### Step 2: Download Dataset

```bash
# Navigate to project directory
cd BigData_FinalProject

# Download Credit Card Fraud Detection dataset
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/

# Unzip dataset
cd data/raw/
unzip creditcardfraud.zip
# Windows: Use 7-Zip or PowerShell: Expand-Archive creditcardfraud.zip

# Verify file
ls -lh creditcard.csv
# Expected: ~150MB file
```

### Step 3: Upload to S3

```bash
# Configure AWS CLI (if not done)
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# Create S3 bucket (replace YOUR_BUCKET_NAME with unique name)
export BUCKET_NAME="csp554-fraud-detection-$(date +%s)"
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Upload dataset
aws s3 cp data/raw/creditcard.csv s3://$BUCKET_NAME/raw-data/creditcard.csv

# Verify upload
aws s3 ls s3://$BUCKET_NAME/raw-data/
```

### Step 4: Schema Validation

**File:** `src/common/schema_validator.py`

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql import functions as F

def get_credit_card_schema():
    """Define expected schema for credit card fraud dataset."""
    return StructType([
        StructField("Time", DoubleType(), True),
        StructField("V1", DoubleType(), True),
        StructField("V2", DoubleType(), True),
        StructField("V3", DoubleType(), True),
        StructField("V4", DoubleType(), True),
        StructField("V5", DoubleType(), True),
        StructField("V6", DoubleType(), True),
        StructField("V7", DoubleType(), True),
        StructField("V8", DoubleType(), True),
        StructField("V9", DoubleType(), True),
        StructField("V10", DoubleType(), True),
        StructField("V11", DoubleType(), True),
        StructField("V12", DoubleType(), True),
        StructField("V13", DoubleType(), True),
        StructField("V14", DoubleType(), True),
        StructField("V15", DoubleType(), True),
        StructField("V16", DoubleType(), True),
        StructField("V17", DoubleType(), True),
        StructField("V18", DoubleType(), True),
        StructField("V19", DoubleType(), True),
        StructField("V20", DoubleType(), True),
        StructField("V21", DoubleType(), True),
        StructField("V22", DoubleType(), True),
        StructField("V23", DoubleType(), True),
        StructField("V24", DoubleType(), True),
        StructField("V25", DoubleType(), True),
        StructField("V26", DoubleType(), True),
        StructField("V27", DoubleType(), True),
        StructField("V28", DoubleType(), True),
        StructField("Amount", DoubleType(), True),
        StructField("Class", IntegerType(), True)  # 0 = Normal, 1 = Fraud
    ])

def validate_dataset(spark, s3_path):
    """Validate dataset schema and basic statistics."""
    schema = get_credit_card_schema()
    df = spark.read.schema(schema).csv(s3_path, header=True)

    # Basic validation
    total_count = df.count()
    fraud_count = df.filter(F.col("Class") == 1).count()
    fraud_rate = (fraud_count / total_count) * 100

    print(f"Total transactions: {total_count:,}")
    print(f"Fraud transactions: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.2f}%")

    # Schema validation
    expected_cols = [field.name for field in schema.fields]
    actual_cols = df.columns
    assert set(expected_cols) == set(actual_cols), f"Schema mismatch: {set(expected_cols) ^ set(actual_cols)}"

    # Missing values check
    null_counts = {col: df.filter(F.col(col).isNull()).count() for col in df.columns}
    print("\nMissing values per column:")
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count}")

    return df

# Usage in EMR Studio
if __name__ == "__main__":
    spark = SparkSession.builder.appName("SchemaValidation").getOrCreate()
    df = validate_dataset(spark, "s3://YOUR_BUCKET_NAME/raw-data/creditcard.csv")
    spark.stop()
```

**Expected Output:**

```
Total transactions: 284,807
Fraud transactions: 492
Fraud rate: 0.17%

Missing values per column:
  (none)
```

---

## Integration Points

### File Paths and Function Names

#### Hussain → Rohit Integration

**Hussain's Output:**

- **File:** `src/hussain/profiling/data_profiler.py`
- **Function:** `generate_profiling_report(spark, s3_input_path, s3_output_path)`
- **Output Location:** `s3://bucket/profiling/report_YYYYMMDD.html`
- **Schema:** Profiling metadata JSON at `s3://bucket/profiling/metadata_YYYYMMDD.json`

**Rohit's Usage:**

```python
# In src/rohit/pipelines/base_pipeline.py
from src.hussain.profiling.data_profiler import get_profiling_metadata

def load_data_with_validation(spark, s3_path):
    metadata = get_profiling_metadata("s3://bucket/profiling/metadata_latest.json")
    # Use metadata for feature selection
    df = spark.read.csv(s3_path, header=True, schema=metadata['schema'])
    return df
```

#### Rohit → Ansh Integration

**Rohit's Output:**

- **File:** `src/rohit/pipelines/base_pipeline.py`
- **Function:** `train_model(spark, train_df, model_type='logistic_regression')`
- **Model Location:** `s3://bucket/models/{model_type}/v{version}/model/`
- **Metadata:** `s3://bucket/models/{model_type}/v{version}/metadata.json`

**Ansh's Usage:**

```python
# In src/ansh/evaluation/metrics.py
from pyspark.ml import PipelineModel

def load_model_for_evaluation(spark, model_path):
    """Load trained model from S3."""
    model = PipelineModel.load(model_path)
    return model

def evaluate_model(model, test_df):
    predictions = model.transform(test_df)
    # Calculate metrics...
    return metrics_dict
```

#### Ansh → Hussain Integration

**Ansh's Output:**

- **File:** `src/ansh/evaluation/metrics.py`
- **Function:** `calculate_all_metrics(predictions_df)`
- **Output Location:** `s3://bucket/outputs/evaluation/{model_name}_metrics.json`
- **Visualizations:** `s3://bucket/outputs/visualizations/{model_name}_roc.png`

**Hussain's Usage:**

```python
# In src/hussain/testing/test_pipelines.py
import json
from src.ansh.evaluation.metrics import load_metrics_from_s3

def validate_model_performance(model_name, min_auroc=0.90):
    metrics = load_metrics_from_s3(f"s3://bucket/outputs/evaluation/{model_name}_metrics.json")
    assert metrics['auroc'] >= min_auroc, f"AUROC {metrics['auroc']} below threshold"
```

#### Shared Utilities

**File:** `src/common/spark_session.py`

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import *

def create_spark_session(app_name="FraudDetection", master="yarn"):
    """Create optimized Spark session for EMR."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark
```

**File:** `src/common/s3_utils.py`

```python
import boto3
from botocore.exceptions import ClientError

def upload_file_to_s3(local_path, s3_bucket, s3_key):
    """Upload file to S3 with error handling."""
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        print(f"Successfully uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        raise

def download_file_from_s3(s3_bucket, s3_key, local_path):
    """Download file from S3 with error handling."""
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(s3_bucket, s3_key, local_path)
        print(f"Successfully downloaded s3://{s3_bucket}/{s3_key} to {local_path}")
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        raise
```

---

## Git Branching Strategy

### Branch Structure

```
main (protected)
├── feature/rohit-pipeline
│   ├── feature/rohit-emr-setup
│   ├── feature/rohit-mllib-pipelines
│   └── feature/rohit-streaming
├── feature/ansh-evaluation
│   ├── feature/ansh-metrics
│   ├── feature/ansh-adversarial
│   └── feature/ansh-streaming-eval
├── feature/hussain-qa
│   ├── feature/hussain-profiling
│   ├── feature/hussain-testing
│   └── feature/hussain-documentation
└── feature/shared-utilities
```

### Branch Naming Convention

- `feature/{owner}-{feature-name}` - Feature branches
- `hotfix/{issue-description}` - Urgent fixes
- `release/v{version}` - Release branches

### Git Workflow Commands

#### Initial Setup

```bash
# Clone repository (after Rohit creates it)
git clone https://github.com/YOUR_USERNAME/BigData_FinalProject.git
cd BigData_FinalProject

# Create and switch to feature branch
git checkout -b feature/rohit-pipeline
# Or for Ansh:
git checkout -b feature/ansh-evaluation
# Or for Hussain:
git checkout -b feature/hussain-qa
```

#### Daily Workflow

```bash
# Start of day: Pull latest changes
git checkout main
git pull origin main
git checkout feature/YOUR-BRANCH
git merge main  # Merge latest main into your branch

# Make changes, then commit
git add src/your-module/
git commit -m "feat: Add evaluation metrics implementation"

# Push to remote
git push origin feature/YOUR-BRANCH
```

#### Merge to Main

```bash
# 1. Ensure your branch is up to date
git checkout feature/YOUR-BRANCH
git merge main

# 2. Run tests (see Quality Gates section)
pytest src/your-module/tests/

# 3. Create Pull Request on GitHub
# 4. After approval, merge via GitHub UI or:
git checkout main
git merge --no-ff feature/YOUR-BRANCH
git push origin main

# 5. Delete feature branch
git branch -d feature/YOUR-BRANCH
git push origin --delete feature/YOUR-BRANCH
```

### Merge Schedule

- **Monday/Wednesday/Friday:** Merge windows (after code review)
- **Before Phase completion:** All feature branches merged to main
- **Emergency:** Hotfix branches can merge anytime with team approval

---

## S3 Bucket Structure

### Complete S3 Path Structure

```
s3://csp554-fraud-detection-{timestamp}/
├── raw-data/
│   └── creditcard.csv                    # Original dataset
├── processed/
│   ├── train/
│   │   └── train_YYYYMMDD.parquet        # Training data
│   ├── test/
│   │   └── test_YYYYMMDD.parquet         # Test data
│   └── validation/
│       └── validation_YYYYMMDD.parquet   # Validation data
├── profiling/
│   ├── report_YYYYMMDD.html               # HTML profiling report
│   ├── metadata_YYYYMMDD.json            # Profiling metadata
│   └── quality_checks_YYYYMMDD.json      # Quality validation results
├── models/
│   ├── logistic_regression/
│   │   ├── v1/
│   │   │   ├── model/                    # Spark ML model files
│   │   │   └── metadata.json             # Model metadata
│   │   └── latest -> v1/                  # Symlink to latest
│   ├── random_forest/
│   │   └── v1/...
│   └── gbt_classifier/
│       └── v1/...
├── outputs/
│   ├── evaluation/
│   │   ├── logistic_regression_metrics.json
│   │   ├── random_forest_metrics.json
│   │   └── gbt_classifier_metrics.json
│   ├── visualizations/
│   │   ├── roc_curves/
│   │   ├── pr_curves/
│   │   └── confusion_matrices/
│   └── adversarial/
│       ├── fgsm_attack_results.json
│       └── robustness_metrics.json
├── checkpoints/
│   ├── streaming/
│   │   └── checkpoint_YYYYMMDD/           # Streaming checkpoints
│   └── training/
│       └── cv_checkpoint_YYYYMMDD/        # Cross-validation checkpoints
└── logs/
    ├── emr-logs/
    └── application-logs/
```

### S3 Access Commands

```bash
# List all files
aws s3 ls s3://$BUCKET_NAME/ --recursive

# Copy file from S3 to local
aws s3 cp s3://$BUCKET_NAME/models/logistic_regression/v1/metadata.json ./local_metadata.json

# Sync directory
aws s3 sync s3://$BUCKET_NAME/outputs/ ./local_outputs/

# Set bucket policy (see IAM section in Rohit's plan)
```

---

## EMR Studio Workspace Organization

### Workspace Structure

```
EMR Studio Workspace: fraud-detection-workspace
├── notebooks/
│   ├── rohit/
│   │   ├── 01_data_loading.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_streaming_integration.ipynb
│   ├── ansh/
│   │   ├── 01_evaluation_metrics.ipynb
│   │   ├── 02_hyperparameter_tuning.ipynb
│   │   ├── 03_adversarial_robustness.ipynb
│   │   └── 04_streaming_evaluation.ipynb
│   └── hussain/
│       ├── 01_data_profiling.ipynb
│       ├── 02_quality_checks.ipynb
│       └── 03_testing_framework.ipynb
└── applications/
    └── fraud-detection-app/
        ├── src/
        └── requirements.txt
```

### Collaboration Rules

1. **Notebook Naming:** `{phase_number}_{description}_{owner}.ipynb`
2. **Cell Documentation:** Every code cell must have markdown explanation above it
3. **Kernel Sharing:** Use shared kernel for consistency
4. **Version Control:** Export notebooks to Git regularly
5. **Resource Limits:** Each notebook session max 2 hours, then restart

### EMR Studio Setup Commands

```bash
# Create EMR Studio (via AWS Console or CLI)
aws emr create-studio \
    --name fraud-detection-workspace \
    --auth-mode SSO \
    --default-s3-location s3://$BUCKET_NAME/emr-studio/ \
    --region us-east-1

# Add users to workspace (IAM users)
aws emr create-studio-session-mapping \
    --studio-id <studio-id> \
    --identity-type USER \
    --identity-id <user-arn>
```

---

## Dependency Matrix

### Task Dependencies

| Task                   | Depends On            | Blocks                 | Owner   |
| ---------------------- | --------------------- | ---------------------- | ------- |
| Dataset Upload         | Kaggle API Setup      | All downstream tasks   | Rohit   |
| Data Profiling         | Dataset Upload        | Feature Engineering    | Hussain |
| Feature Engineering    | Data Profiling        | Model Training         | Rohit   |
| Model Training         | Feature Engineering   | Model Evaluation       | Rohit   |
| Model Evaluation       | Model Training        | Adversarial Testing    | Ansh    |
| Adversarial Testing    | Model Evaluation      | Defense Implementation | Ansh    |
| Defense Implementation | Adversarial Testing   | Final Testing          | Hussain |
| Streaming Integration  | Model Training        | Streaming Evaluation   | Rohit   |
| Streaming Evaluation   | Streaming Integration | Final Report           | Ansh    |
| Final Testing          | All components        | Documentation          | Hussain |
| Documentation          | All components        | Submission             | Hussain |

### Critical Path

```
Dataset Upload → Data Profiling → Feature Engineering → Model Training
→ Model Evaluation → Adversarial Testing → Defense Implementation
→ Streaming Integration → Final Testing → Documentation
```

**Estimated Critical Path Duration:** 18 days (with buffer)

---

## Cost Management

### AWS Cost Estimates

| Phase     | Component                          | Estimated Cost         | Duration    |
| --------- | ---------------------------------- | ---------------------- | ----------- |
| Phase 0   | S3 Storage (150MB)                 | $0.003/month           | Ongoing     |
| Phase 1   | EMR Cluster (3x m5.xlarge)         | $0.192/hour × 20 hours | $3.84       |
| Phase 2   | EMR Cluster + S3 Operations        | $0.192/hour × 30 hours | $5.76       |
| Phase 3   | EMR Cluster + Adversarial Training | $0.192/hour × 25 hours | $4.80       |
| Phase 4   | EMR Cluster + Streaming            | $0.192/hour × 15 hours | $2.88       |
| Phase 5   | EMR Cluster (minimal)              | $0.192/hour × 5 hours  | $0.96       |
| **Total** |                                    |                        | **~$18.24** |

_Note: Costs vary by region and usage. Monitor actual costs in AWS Cost Explorer._

### Cost Optimization Checklist

- [ ] Use Spot Instances for non-critical workloads (save 70%)
- [ ] Terminate cluster immediately after use
- [ ] Use S3 Intelligent-Tiering for long-term storage
- [ ] Enable EMR auto-scaling only when needed
- [ ] Set up billing alerts at $10, $15, $20 thresholds
- [ ] Use EMR Serverless for batch jobs (if available)
- [ ] Compress data before S3 upload (Parquet format)

### AWS Budget Alert Setup

```bash
# Create budget alert
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget file://budget-config.json \
    --notifications-with-subscribers file://budget-notifications.json
```

**budget-config.json:**

```json
{
  "BudgetName": "fraud-detection-project-budget",
  "BudgetLimit": {
    "Amount": "25",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

**budget-notifications.json:**

```json
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "team-email@example.com"
      }
    ]
  }
]
```

### Cost Monitoring Commands

```bash
# Check current month costs
aws ce get-cost-and-usage \
    --time-period Start=2025-11-01,End=2025-12-01 \
    --granularity MONTHLY \
    --metrics BlendedCost

# Check EMR costs specifically
aws ce get-cost-and-usage \
    --time-period Start=2025-11-01,End=2025-12-01 \
    --granularity DAILY \
    --metrics BlendedCost \
    --filter file://emr-filter.json
```

---

## Shared Utilities

### Directory Structure

```
src/common/
├── __init__.py
├── spark_session.py          # Spark session creation
├── s3_utils.py               # S3 operations
├── schema_validator.py       # Dataset schema validation
├── config.py                 # Configuration management
└── logger.py                 # Logging utilities
```

### Configuration File

**File:** `src/common/config.py`

```python
import os
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    """Centralized configuration for the project."""
    # S3 Configuration
    S3_BUCKET: str = os.getenv("S3_BUCKET", "csp554-fraud-detection-default")
    S3_RAW_DATA_PATH: str = f"s3://{S3_BUCKET}/raw-data/creditcard.csv"
    S3_PROCESSED_PATH: str = f"s3://{S3_BUCKET}/processed/"
    S3_MODELS_PATH: str = f"s3://{S3_BUCKET}/models/"
    S3_OUTPUTS_PATH: str = f"s3://{S3_BUCKET}/outputs/"

    # Spark Configuration
    SPARK_APP_NAME: str = "FraudDetection"
    SPARK_EXECUTOR_MEMORY: str = "4g"
    SPARK_EXECUTOR_CORES: int = 2
    SPARK_DRIVER_MEMORY: str = "2g"

    # Model Configuration
    RANDOM_SEED: int = 42
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.1

    # Evaluation Configuration
    MIN_AUROC_THRESHOLD: float = 0.90

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            S3_BUCKET=os.getenv("S3_BUCKET", cls.S3_BUCKET),
            RANDOM_SEED=int(os.getenv("RANDOM_SEED", cls.RANDOM_SEED))
        )

# Global config instance
config = ProjectConfig.from_env()
```

**Usage:**

```python
from src.common.config import config

# Use config throughout project
df = spark.read.csv(config.S3_RAW_DATA_PATH, header=True)
```

---

## Conflict Resolution Protocols

### Shared File Conflict Resolution

#### Scenario 1: Conflict in `src/common/`

**Process:**

1. Identify conflicting changes
2. Team meeting (15 min) to discuss both implementations
3. Choose best approach or merge both
4. Document decision in `CONFLICT_RESOLUTION_LOG.md`

**Example:**

```bash
# When conflict occurs
git pull origin main
# CONFLICT: src/common/spark_session.py

# Resolution steps:
# 1. Open conflict file
# 2. Discuss with team
# 3. Manually merge or choose one version
# 4. Test merged version
git add src/common/spark_session.py
git commit -m "resolve: Merge spark_session.py configurations"
```

#### Scenario 2: Schema Changes

**Process:**

1. Notify team via Slack/email before schema changes
2. Update `src/common/schema_validator.py` first
3. Wait for team acknowledgment (24 hours)
4. Proceed with changes

#### Scenario 3: API Contract Changes

**Process:**

1. Document breaking changes in `CHANGELOG.md`
2. Maintain backward compatibility for 1 week
3. Deprecate old API with warnings
4. Remove after team migration

### Communication Protocols

- **Daily Standup:** 9:00 AM (15 minutes)
- **Blockers:** Immediate Slack message to team
- **Merge Conflicts:** Schedule 30-min resolution meeting
- **Critical Issues:** Emergency call (all members)

---

## Daily Standup Agenda

### Format (15 minutes max)

1. **What did you complete yesterday?** (2 min each)
2. **What are you working on today?** (2 min each)
3. **Any blockers?** (1 min each)
4. **Integration points update** (2 min total)

### Standup Template

```markdown
## Daily Standup - {Date}

### Ansh

- **Yesterday:** Completed AUROC/AUPRC implementation
- **Today:** Working on hyperparameter tuning script
- **Blockers:** None
- **Integration:** Waiting for Rohit's model v1

### Rohit

- **Yesterday:** Set up EMR cluster, created base pipeline
- **Today:** Implementing Logistic Regression pipeline
- **Blockers:** Need Hussain's profiling output for feature selection
- **Integration:** Will push model to S3 by EOD

### Hussain

- **Yesterday:** Completed data profiling script
- **Today:** Generating profiling report, running quality checks
- **Blockers:** None
- **Integration:** Profiling report will be ready by 2 PM
```

### Progress Tracking

**File:** `PROGRESS_TRACKER.md` (updated daily)

```markdown
## Progress Tracker

### Phase 1 (Nov 18-22)

- [x] Dataset uploaded to S3
- [x] Data profiling completed
- [ ] Feature engineering (In Progress - Rohit)
- [ ] Model training (Pending - Rohit)
- [ ] Evaluation framework (In Progress - Ansh)

### Phase 2 (Nov 23-27)

- [ ] Hyperparameter tuning
- [ ] Model comparison
- [ ] Visualization generation
```

---

## Code Review Checklist

### Pre-Merge Checklist

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Unit tests pass (`pytest src/your-module/tests/`)
- [ ] Integration tests pass (if applicable)
- [ ] No hardcoded credentials or paths
- [ ] Error handling implemented
- [ ] Logging added for critical operations
- [ ] Performance considerations documented
- [ ] Dependencies added to `requirements.txt`
- [ ] README updated (if new features)

### Code Review Process

1. **Author:** Create PR with description
2. **Reviewer 1:** Review within 24 hours
3. **Reviewer 2:** Review within 24 hours (if major changes)
4. **Approval:** 2 approvals required for merge
5. **Merge:** Author or maintainer merges after approval

### Review Focus Areas

- **Functionality:** Does it work as intended?
- **Performance:** Any obvious bottlenecks?
- **Security:** No exposed credentials or vulnerabilities
- **Maintainability:** Is code readable and well-documented?
- **Integration:** Does it break existing integrations?

---

## Emergency Scenarios & Rollback Procedures

### Scenario 1: EMR Cluster Termination

**Symptoms:** Cluster unexpectedly terminated, work lost

**Recovery:**

```bash
# 1. Check CloudWatch logs
aws logs tail /aws/emr/cluster-name --follow

# 2. Recreate cluster with same configuration
# (Use Rohit's emr_setup.sh script)

# 3. Restore data from S3 (if needed)
aws s3 sync s3://$BUCKET_NAME/checkpoints/ ./local_checkpoints/

# 4. Resume from last checkpoint
```

**Prevention:**

- Save notebooks to Git regularly
- Use S3 checkpoints for long-running jobs
- Enable EMR auto-termination protection

### Scenario 2: Data Corruption

**Symptoms:** Unexpected results, schema mismatches

**Recovery:**

```bash
# 1. Verify original dataset
aws s3 ls s3://$BUCKET_NAME/raw-data/ --recursive

# 2. Re-download from Kaggle if needed
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/

# 3. Re-upload to S3
aws s3 cp data/raw/creditcard.csv s3://$BUCKET_NAME/raw-data/creditcard.csv

# 4. Re-run profiling and processing
```

### Scenario 3: Merge Conflict Escalation

**Symptoms:** Complex conflicts that can't be resolved

**Recovery:**

```bash
# 1. Create backup branch
git checkout -b backup/feature-name-$(date +%Y%m%d)

# 2. Reset to last known good state
git checkout feature/your-branch
git reset --hard origin/main

# 3. Re-apply changes manually (cherry-pick if possible)
git cherry-pick <commit-hash>

# 4. Test thoroughly before merge
```

### Scenario 4: Budget Exceeded

**Symptoms:** AWS budget alert triggered

**Recovery:**

1. Immediately terminate all EMR clusters
2. Switch to local Spark development (Plan B)
3. Use smaller datasets for testing
4. Resume EMR only for final runs

**Plan B: Local Spark Setup**

```bash
# Install local Spark
pip install pyspark

# Run with local master
spark = SparkSession.builder \
    .appName("LocalFraudDetection") \
    .master("local[*]") \
    .getOrCreate()

# Use local file paths instead of S3
df = spark.read.csv("data/raw/creditcard.csv", header=True)
```

### Scenario 5: Deadline Pressure

**Symptoms:** Behind schedule, critical features incomplete

**Recovery:**

1. **Prioritize:** Focus on core deliverables (models, evaluation, report)
2. **Simplify:** Remove optional features (streaming if not critical)
3. **Document:** Clearly mark what's complete vs. future work
4. **Communicate:** Notify professor early about scope adjustments

---

## Deliverable Checklist

### Mapping to Grading Criteria

| Deliverable      | Grading Weight | Owner                          | Status |
| ---------------- | -------------- | ------------------------------ | ------ |
| Technical Report | 40%            | Hussain (lead), All contribute | [ ]    |
| Working Demo     | 25%            | Rohit (lead), Ansh supports    | [ ]    |
| Code Repository  | 20%            | All (Rohit maintains)          | [ ]    |
| Presentation     | 15%            | All (equal time)               | [ ]    |

### Technical Report Sections

- [ ] **Introduction** (Hussain)
  - Problem statement
  - Dataset description
  - Objectives
- [ ] **Literature Review** (Hussain)
  - Fraud detection methods
  - Adversarial robustness in ML
  - Spark MLlib applications
- [ ] **Methodology** (Rohit, Ansh)
  - Data preprocessing
  - Model architectures
  - Evaluation metrics
  - Adversarial attack methods
- [ ] **Results** (Ansh, Hussain)
  - Model performance comparison
  - Adversarial robustness analysis
  - Visualization figures
- [ ] **Discussion** (All)
  - Findings interpretation
  - Limitations
  - Future work
- [ ] **Conclusion** (Hussain)
- [ ] **References** (Hussain - BibTeX format)

### Code Repository Requirements

- [ ] Clean, organized structure
- [ ] Comprehensive README.md
- [ ] Requirements.txt with versions
- [ ] Setup instructions
- [ ] Example usage scripts
- [ ] Documentation strings
- [ ] Test coverage > 80%

### Presentation Requirements

- [ ] 15-20 minute presentation
- [ ] 5 minutes per team member
- [ ] Live demo (or recorded)
- [ ] Q&A preparation
- [ ] Backup slides for deep dives

### Grader's Perspective Verification

**Before Submission, Verify:**

```bash
# 1. Can grader clone and run?
git clone <repo-url>
cd BigData_FinalProject
pip install -r requirements.txt
# Follow README instructions
# Does it work?

# 2. Are all paths configurable?
# Check: No hardcoded S3 paths, use config.py

# 3. Are results reproducible?
# Check: Random seeds set, deterministic algorithms

# 4. Is documentation complete?
# Check: README, docstrings, comments

# 5. Are deliverables accessible?
# Check: Report PDF, presentation slides, demo video
```

---

## Final Integration & Testing Protocol

### Pre-Submission Checklist (Dec 9, 2025)

#### Code Integration

- [ ] All feature branches merged to main
- [ ] No merge conflicts
- [ ] All tests pass
- [ ] Code follows style guide
- [ ] No TODO comments left

#### Data Pipeline Integration

- [ ] End-to-end pipeline runs successfully
- [ ] Data flows: Raw → Profiled → Processed → Trained → Evaluated
- [ ] All S3 paths accessible
- [ ] Models loadable from S3

#### Model Integration

- [ ] All three models (LR, RF, GBT) trained and saved
- [ ] Evaluation metrics calculated for all models
- [ ] Adversarial robustness tested
- [ ] Streaming integration working (if implemented)

#### Documentation Integration

- [ ] Technical report complete
- [ ] All figures and tables included
- [ ] Code repository documented
- [ ] Presentation slides finalized

### Final Testing Script

**File:** `scripts/final_integration_test.py`

```python
"""
Final integration test to verify all components work together.
Run this before submission.
"""
import sys
from pyspark.sql import SparkSession
from src.common.config import config
from src.common.spark_session import create_spark_session
from src.hussain.profiling.data_profiler import validate_data_quality
from src.rohit.pipelines.base_pipeline import load_and_train_model
from src.ansh.evaluation.metrics import evaluate_model_comprehensive

def test_end_to_end_pipeline():
    """Test complete pipeline from data loading to evaluation."""
    spark = create_spark_session("IntegrationTest")

    try:
        # 1. Data loading
        print("Step 1: Loading data...")
        df = spark.read.csv(config.S3_RAW_DATA_PATH, header=True)
        assert df.count() > 0, "Data loading failed"
        print(f"✓ Loaded {df.count():,} records")

        # 2. Data profiling
        print("Step 2: Running data profiling...")
        quality_report = validate_data_quality(df)
        assert quality_report['fraud_rate'] > 0, "No fraud cases found"
        print(f"✓ Fraud rate: {quality_report['fraud_rate']:.2f}%")

        # 3. Model training
        print("Step 3: Training model...")
        model = load_and_train_model(spark, df, model_type='logistic_regression')
        assert model is not None, "Model training failed"
        print("✓ Model trained successfully")

        # 4. Model evaluation
        print("Step 4: Evaluating model...")
        metrics = evaluate_model_comprehensive(model, df)
        assert metrics['auroc'] >= config.MIN_AUROC_THRESHOLD, \
            f"AUROC {metrics['auroc']} below threshold {config.MIN_AUROC_THRESHOLD}"
        print(f"✓ AUROC: {metrics['auroc']:.4f}")

        print("\n✅ All integration tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        spark.stop()

if __name__ == "__main__":
    success = test_end_to_end_pipeline()
    sys.exit(0 if success else 1)
```

### Run Final Tests

```bash
# 1. Run integration tests
python scripts/final_integration_test.py

# 2. Run unit tests
pytest src/ --cov=src --cov-report=html

# 3. Verify S3 structure
aws s3 ls s3://$BUCKET_NAME/ --recursive > s3_structure.txt

# 4. Generate final report
python scripts/generate_final_report.py

# 5. Create submission package
./scripts/create_submission_package.sh
```

### Submission Package Structure

```
submission/
├── technical_report.pdf
├── presentation_slides.pdf
├── demo_video.mp4 (optional)
├── code_repository.zip (or GitHub link)
└── README_SUBMISSION.md
```

---

## Emergency Contacts & Escalation

### Team Contacts

- **Ansh Kaushik:** [Email] | [Phone] | Slack: @ansh
- **Rohit Lahori:** [Email] | [Phone] | Slack: @rohit
- **Hussain Bin Yousuf:** [Email] | [Phone] | Slack: @hussain

### Escalation Path

1. **Level 1:** Team discussion (Slack/meeting)
2. **Level 2:** Professor consultation (CSP 554)
3. **Level 3:** AWS Support (for infrastructure issues)
4. **Level 4:** Academic advisor (for timeline issues)

### AWS Support

- **Basic Support:** Included (email support)
- **Developer Support:** $29/month (if needed)
- **Support Center:** https://console.aws.amazon.com/support/

### Critical Issue Response Time

- **Data Loss:** Immediate (all hands)
- **Cluster Failure:** < 2 hours
- **Merge Conflicts:** < 4 hours
- **Documentation Issues:** < 24 hours

---

## Pro Tips

1. **Use Parquet Format:** Convert CSV to Parquet for 10x faster reads
2. **Cache Strategically:** Cache DataFrames used multiple times
3. **Monitor Resources:** Use CloudWatch to track cluster utilization
4. **Version Everything:** Tag Git commits and S3 model versions
5. **Document Decisions:** Keep a decision log for future reference
6. **Test Early:** Run integration tests after each major merge
7. **Backup Regularly:** Export notebooks and save to Git daily
8. **Optimize Costs:** Use Spot Instances for non-critical workloads
9. **Use Checkpoints:** Save intermediate results to S3
10. **Communicate Often:** Daily standups prevent integration issues

---

## Appendix

### A. Useful AWS CLI Commands

```bash
# List EMR clusters
aws emr list-clusters --active

# Describe cluster
aws emr describe-cluster --cluster-id <cluster-id>

# Terminate cluster
aws emr terminate-clusters --cluster-ids <cluster-id>

# View logs
aws s3 ls s3://aws-logs-<account-id>-<region>/elasticmapreduce/<cluster-id>/
```

### B. Useful Git Commands

```bash
# View commit history
git log --oneline --graph --all

# Find large files
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print substr($0,6)}' | sort --numeric-sort --key=2 | tail -10

# Clean up branches
git branch --merged | grep -v "\*\|main" | xargs -n 1 git branch -d
```

### C. Useful Spark Commands

```bash
# Submit Spark job
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 4g \
    --executor-cores 2 \
    --num-executors 3 \
    src/your_script.py

# View Spark UI
# Access via EMR Studio or: http://<master-node>:8088
```

---

**Document Version:** 1.0  
**Last Updated:** November 18, 2025  
**Next Review:** November 25, 2025
