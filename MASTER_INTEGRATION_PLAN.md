# Master Integration Plan

## CSP 554 Big Data Technologies - Fraud Detection Project

**Team Members:** Ansh Kaushik, Rohit Lahori, Hussain Bin Yousuf  
**Timeline:** 2-Day Sprint Timeline (16-20 working hours)  
**Platform:** AWS EMR with Apache Spark MLlib  
**Dataset:** creditcard_2023.csv (Credit Card Fraud Detection dataset)  
**Development Approach:** Cursor-assisted rapid development for maximum efficiency

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [2-Day Execution Workflow](#2-day-execution-workflow)
3. [Pre-Day 1: Dataset Preparation](#pre-day-1-dataset-preparation)
4. [Integration Points](#integration-points)
5. [Git Branching Strategy](#git-branching-strategy)
6. [S3 Bucket Structure](#s3-bucket-structure)
7. [EMR Studio Workspace Organization](#emr-studio-workspace-organization)
8. [Dependency Matrix](#dependency-matrix)
9. [Cost Management](#cost-management)
10. [Shared Utilities](#shared-utilities)
11. [Conflict Resolution Protocols](#conflict-resolution-protocols)
12. [Daily Standup Agenda](#daily-standup-agenda)
13. [Code Review Checklist](#code-review-checklist)
14. [Emergency Scenarios & Rollback Procedures](#emergency-scenarios--rollback-procedures)
15. [Deliverable Checklist](#deliverable-checklist)
16. [Final Integration & Testing Protocol](#final-integration--testing-protocol)

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

## 2-Day Execution Workflow

### Overview

**Total Timeline:** 2 days (16-20 working hours)  
**EMR Cluster Sessions:** 3 sessions (~8-10 hours total)  
**Cost Target:** <$5 with optimization  
**Strategy:** Maximize parallel work, minimize EMR cluster time, use Cursor for rapid development

### Workflow Timeline Diagram

```
Day 1 (Hours 0-10)                          Day 2 (Hours 10-20)
┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐
│ H0-1: Setup (All)                   │    │ H10-12: Adversarial (Ansh+Hussain) │
│ H1-2: Shared Utils (Parallel)       │    │ H12-14: Streaming (Rohit+Ansh)      │
│ H2-4: Profiling (Hussain) [EMR-1]   │    │ H14-16: Integration (All)           │
│ H4-6: Features (Rohit) [EMR-2]      │    │ H16-18: Documentation (Hussain)     │
│ H6-8: Training (Rohit+Ansh) [EMR-2] │    │ H18-20: Final Polish (All)          │
│ H8-10: Tuning (Ansh)                │    │                                     │
└─────────────────────────────────────┘    └─────────────────────────────────────┘
```

### Day 1: Foundation & Core Development (Hours 0-10)

#### Hour 0-1: Initial Setup (All Together)

**Goal:** Get all infrastructure and local environment ready

**Rohit (Technical Lead):**

- Create S3 bucket
  ```bash
  export BUCKET_NAME="csp554-fraud-detection-$(date +%s)"
  aws s3 mb s3://$BUCKET_NAME --region us-east-1
  ```
- Upload creditcard_2023.csv to S3
  ```bash
  aws s3 cp creditcard_2023.csv/creditcard_2023.csv s3://$BUCKET_NAME/raw-data/creditcard_2023.csv
  ```
- Prepare EMR cluster configuration (don't start yet)
  - Create `src/rohit/infrastructure/emr-config.json`
  - Use m5.large instances (cheaper)
  - Configure auto-termination

**Hussain (QA Lead):**

- Validate dataset locally
  ```python
  # Quick validation script
  import pandas as pd
  df_sample = pd.read_csv('creditcard_2023.csv/creditcard_2023.csv', nrows=1000)
  print(f"Columns: {df_sample.columns.tolist()}")
  print(f"Shape: {df_sample.shape}")
  print(f"Missing values: {df_sample.isnull().sum().sum()}")
  if 'Class' in df_sample.columns:
      print(f"Fraud rate: {df_sample['Class'].mean():.4%}")
  ```
- Document dataset schema and characteristics

**Ansh (Evaluation Lead):**

- Set up project structure
  ```bash
  mkdir -p src/{common,rohit,ansh,hussain}/{__pycache__}
  mkdir -p tests/{rohit,ansh,hussain}
  mkdir -p notebooks/{rohit,ansh,hussain}
  ```
- Install dependencies locally
  ```bash
  pip install pyspark numpy pandas matplotlib seaborn scikit-learn mlflow boto3 pyarrow
  ```

**Deliverable:** All infrastructure ready, dataset validated, local environments set up

**Integration Point:** Team sync at end of Hour 1 to confirm S3 bucket name and dataset location

---

#### Hour 1-2: Shared Utilities (Parallel Development)

**Goal:** Create common utilities that all team members will use

**Rohit:**

- Create `src/common/spark_session.py`

  ```python
  from pyspark.sql import SparkSession

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

- Create `src/common/config.py`

  ```python
  import os
  from dataclasses import dataclass

  @dataclass
  class ProjectConfig:
      S3_BUCKET: str = os.getenv("S3_BUCKET", "csp554-fraud-detection-default")
      RAW_DATA_PATH: str = f"s3://{S3_BUCKET}/raw-data/creditcard_2023.csv"
      PROCESSED_DATA_PATH: str = f"s3://{S3_BUCKET}/processed/"
      MODELS_PATH: str = f"s3://{S3_BUCKET}/models/"
      OUTPUTS_PATH: str = f"s3://{S3_BUCKET}/outputs/"

      @classmethod
      def from_env(cls):
          return cls(S3_BUCKET=os.getenv("S3_BUCKET", "csp554-fraud-detection-default"))

  config = ProjectConfig.from_env()
  ```

**Hussain:**

- Create `src/common/schema_validator.py`

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
          # ... V3-V28 ...
          StructField("Amount", DoubleType(), True),
          StructField("Class", IntegerType(), True),
      ])

  def validate_dataset(spark, s3_path):
      """Validate dataset schema and basic quality."""
      df = spark.read.csv(s3_path, header=True)
      # Add validation logic
      return df
  ```

**Ansh:**

- Create `src/common/s3_utils.py`

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

**Git Workflow:**

```bash
# Each person works on their branch
git checkout -b feature/shared-utilities
# After completing their file:
git add src/common/YOUR_FILE.py
git commit -m "feat: Add [file] to shared utilities"
git push origin feature/shared-utilities

# Rohit merges all to main
git checkout main
git merge feature/shared-utilities
git push origin main
```

**Deliverable:** All shared utilities created and merged to main branch

**Integration Checkpoint 1 (Hour 2):** Shared utilities merged, all team members pull latest main

---

#### Hour 2-4: Data Profiling (Hussain) + Code Preparation (Rohit & Ansh)

**Goal:** Complete data profiling on EMR, prepare feature engineering and evaluation code locally

**Hussain (EMR Session 1 - 2 hours):**

- Start EMR cluster (Rohit helps with cluster creation)
  ```bash
  aws emr create-cluster \
      --name fraud-detection-profiling \
      --release-label emr-6.15.0 \
      --instance-type m5.large \
      --instance-count 3 \
      --applications Name=Spark Name=JupyterEnterpriseGateway \
      --auto-terminate \
      --ec2-attributes KeyName=YOUR_KEY_NAME
  ```
- Run data profiling on full dataset

  ```python
  # In EMR Studio notebook: notebooks/hussain/01_data_profiling.ipynb
  from src.common.spark_session import create_spark_session
  from src.common.config import config
  from src.hussain.profiling.data_profiler import CreditCardDataProfiler

  spark = create_spark_session("DataProfiling")
  profiler = CreditCardDataProfiler(spark)

  df = spark.read.csv(config.RAW_DATA_PATH, header=True)
  profile = profiler.profile_dataset(df)
  profiler.save_profile_to_s3(profile, f"s3://{config.S3_BUCKET}/profiling/")
  ```

- Generate quality report and save to S3
- **Terminate cluster immediately after profiling completes**

**Rohit (Local Development):**

- Write feature engineering pipeline code locally
  - Test with small sample dataset
  - Create `src/rohit/pipelines/base_pipeline.py`
  - Implement VectorAssembler, StandardScaler stages

**Ansh (Local Development):**

- Write evaluation metrics code locally
  - Create `src/ansh/evaluation/metrics.py`
  - Implement AUROC, AUPRC, Precision, Recall, F1
  - Test with mock predictions

**Deliverable:** Profiling report in S3, feature engineering and evaluation code ready for EMR

**Handoff Point:** Hussain provides profiling report location → Rohit uses insights for feature engineering

**Integration Checkpoint 2 (Hour 4):** Profiling complete, feature engineering can start

---

#### Hour 4-6: Feature Engineering & Model Training Prep (Rohit) + Evaluation Framework (Ansh)

**Goal:** Process data and prepare for model training

**Rohit (EMR Session 2 - Start, continues to Hour 8):**

- Start EMR cluster (2nd session)
- Load profiling metadata from Hussain's output
  ```python
  # Load profiling insights
  from src.hussain.profiling.data_profiler import get_profiling_metadata
  metadata = get_profiling_metadata(f"s3://{config.S3_BUCKET}/profiling/metadata_latest.json")
  ```
- Run feature engineering pipeline

  ```python
  from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline

  pipeline = BaseFraudDetectionPipeline(spark)
  df = pipeline.load_data(config.RAW_DATA_PATH)
  feature_pipeline = pipeline.create_feature_pipeline()
  processed_df = feature_pipeline.fit(df).transform(df)

  # Save processed data
  processed_df.write.parquet(config.PROCESSED_DATA_PATH + "processed_data.parquet")
  ```

- Prepare train/test split
  ```python
  train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
  train_df.write.parquet(config.PROCESSED_DATA_PATH + "train/")
  test_df.write.parquet(config.PROCESSED_DATA_PATH + "test/")
  ```

**Ansh (Local Development):**

- Complete evaluation framework
  - Finish `src/ansh/evaluation/metrics.py`
  - Create `src/ansh/evaluation/cross_validation.py`
  - Test with sample data locally

**Hussain (Local Development):**

- Write test cases
  - Create `tests/test_pipelines.py`
  - Create `tests/test_evaluation.py`
- Start documentation structure

**Deliverable:** Processed data in S3, evaluation framework ready, test cases written

---

#### Hour 6-8: Model Training (Rohit) + Model Evaluation (Ansh)

**Goal:** Train all three models and evaluate them

**Rohit (Continues EMR Session 2):**

- Train Logistic Regression

  ```python
  from src.rohit.pipelines.logistic_regression import LogisticRegressionPipeline

  lr_pipeline = LogisticRegressionPipeline(spark)
  lr_model = lr_pipeline.train(train_df)
  lr_pipeline.save_model(lr_model, "logistic_regression", "v1")
  ```

- Train Random Forest

  ```python
  from src.rohit.pipelines.random_forest import RandomForestPipeline

  rf_pipeline = RandomForestPipeline(spark)
  rf_model = rf_pipeline.train(train_df)
  rf_pipeline.save_model(rf_model, "random_forest", "v1")
  ```

- Train GBT Classifier

  ```python
  from src.rohit.pipelines.gbt_classifier import GBTPipeline

  gbt_pipeline = GBTPipeline(spark)
  gbt_model = gbt_pipeline.train(train_df)
  gbt_pipeline.save_model(gbt_model, "gbt_classifier", "v1")
  ```

**Ansh (Parallel on EMR):**

- Evaluate each model as it's trained

  ```python
  from src.ansh.evaluation.metrics import FraudDetectionMetrics

  metrics_calc = FraudDetectionMetrics(spark)

  # Evaluate LR
  lr_predictions = lr_model.transform(test_df)
  lr_metrics = metrics_calc.calculate_all_metrics(lr_predictions)
  metrics_calc.save_metrics_to_s3(lr_metrics, "logistic_regression", "v1")

  # Repeat for RF and GBT
  ```

**Hussain (Local):**

- Run integration tests
- Update documentation

**Deliverable:** All three models trained and evaluated, metrics saved to S3

**Integration Checkpoint 3 (Hour 8):** Models trained, evaluation complete, ready for hyperparameter tuning

---

#### Hour 8-10: Hyperparameter Tuning (Ansh) + Testing (Hussain) + Optimization (Rohit)

**Goal:** Optimize models and ensure quality

**Ansh (EMR Session 2 - Continue):**

- Hyperparameter tuning with MLflow

  ```python
  from src.ansh.evaluation.hyperparameter_tuning import HyperparameterTuner
  from pyspark.ml.classification import LogisticRegression

  tuner = HyperparameterTuner(spark)
  best_lr_model = tuner.tune_logistic_regression(train_df, test_df)
  # Repeat for RF and GBT
  ```

- Generate comparison report

  ```python
  from src.ansh.evaluation.model_comparison import ModelComparator

  comparator = ModelComparator(spark)
  comparison_report = comparator.compare_models(
      [lr_metrics, rf_metrics, gbt_metrics],
      ["Logistic Regression", "Random Forest", "GBT"]
  )
  comparator.save_report_to_s3(comparison_report)
  ```

**Hussain (Local):**

- Comprehensive testing
  ```bash
  pytest tests/ -v --cov=src
  ```
- Data quality validation
- Update test coverage

**Rohit (Local):**

- Model optimization review
- Prepare for adversarial training integration
- **Terminate EMR cluster after Ansh completes tuning**

**Deliverable:** Optimized models, comprehensive test coverage, ready for Day 2

**End of Day 1 Checkpoint (Hour 10):** All core components complete, models optimized, tests passing

---

### Day 2: Advanced Features & Integration (Hours 10-20)

#### Hour 10-12: Adversarial Robustness (Ansh + Hussain)

**Goal:** Implement and test adversarial attacks and defenses

**Ansh (Local Prep, then EMR Session 3):**

- Implement FGSM attack (local prep)

  ```python
  # src/ansh/adversarial/fgsm_attack.py
  from src.ansh.adversarial.fgsm_attack import FGSMAttack

  attack = FGSMAttack(epsilon=0.1)
  adversarial_examples = attack.generate_attacks(model, test_df)
  ```

- Test locally with sample data

**Hussain (Local Prep, then EMR Session 3):**

- Implement defense mechanisms (local prep)

  ```python
  # src/hussain/defenses/feature_clamping.py
  from src.hussain.defenses.feature_clamping import FeatureClamping

  defense = FeatureClamping(min_values, max_values)
  protected_df = defense.apply(test_df)
  ```

- Test locally

**Rohit (EMR Session 3 - Start):**

- Start EMR cluster (3rd session)
- Integrate adversarial training pipeline

  ```python
  from src.rohit.pipelines.adversarial_training import AdversarialTrainingPipeline

  adv_pipeline = AdversarialTrainingPipeline(spark)
  robust_model = adv_pipeline.train_with_adversarial(train_df, adversarial_examples)
  ```

**All Together (EMR Session 3):**

- Run adversarial attacks on all models
- Test defense mechanisms
- Calculate robustness metrics

  ```python
  from src.ansh.adversarial.robustness_metrics import RobustnessEvaluator

  evaluator = RobustnessEvaluator(spark)
  robustness_scores = evaluator.evaluate_robustness(
      models=[lr_model, rf_model, gbt_model],
      test_df=test_df,
      adversarial_examples=adversarial_examples
  )
  evaluator.save_robustness_metrics_to_s3(robustness_scores)
  ```

**Deliverable:** Adversarial attacks implemented, defenses tested, robustness metrics calculated

---

#### Hour 12-14: Streaming Integration (Rohit + Ansh)

**Goal:** Implement structured streaming for real-time fraud detection

**Rohit (Continues EMR Session 3):**

- Create streaming pipeline

  ```python
  from src.rohit.streaming.streaming_integration import StreamingFraudDetection

  streaming_app = StreamingFraudDetection(spark, best_model)
  streaming_query = streaming_app.start_streaming(
      input_path="s3://bucket/streaming-input/",
      output_path="s3://bucket/streaming-output/",
      checkpoint_path="s3://bucket/checkpoints/streaming/"
  )
  ```

**Ansh (Parallel on EMR):**

- Implement streaming evaluation

  ```python
  from src.ansh.streaming.streaming_evaluator import StreamingEvaluator

  evaluator = StreamingEvaluator(spark)
  streaming_metrics = evaluator.evaluate_streaming_batches(
      streaming_output_path,
      batch_interval="1 minute"
  )
  ```

**Hussain (Local):**

- Test streaming components
- Update documentation

**Deliverable:** Streaming pipeline operational, streaming evaluation working

---

#### Hour 14-16: Final Integration & Testing (All)

**Goal:** End-to-end integration and comprehensive testing

**All Together:**

- Run complete end-to-end pipeline

  ```python
  # scripts/final_integration_test.py
  from src.common.spark_session import create_spark_session
  from src.common.config import config

  spark = create_spark_session("FinalIntegration")

  # Test data loading
  df = spark.read.csv(config.RAW_DATA_PATH, header=True)
  assert df.count() > 0, "Data loading failed"

  # Test model loading
  from src.rohit.pipelines.base_pipeline import load_model_from_s3
  model = load_model_from_s3("logistic_regression", "v1")
  assert model is not None, "Model loading failed"

  # Test evaluation
  predictions = model.transform(test_df)
  from src.ansh.evaluation.metrics import FraudDetectionMetrics
  metrics = FraudDetectionMetrics(spark).calculate_all_metrics(predictions)
  assert metrics['auroc'] > 0.85, "Model performance below threshold"

  print("✅ All integration tests passed!")
  ```

- Performance benchmarking
- Save all final outputs to S3
- **Terminate EMR cluster**

**Deliverable:** Fully integrated system, all tests passing, performance benchmarks documented

**Integration Checkpoint 5 (Hour 16):** Advanced features complete, system fully integrated

---

#### Hour 16-18: Documentation & Report (Hussain Lead, All Contribute)

**Goal:** Complete technical report and documentation

**Hussain (Lead):**

- Technical report structure
  - Introduction
  - Methodology
  - Results
  - Discussion
  - Conclusion
- Code documentation
- Presentation outline

**Ansh:**

- Results section with visualizations
- Metrics interpretation
- Adversarial robustness analysis

**Rohit:**

- Architecture documentation
- Pipeline descriptions
- Performance analysis

**All:**

- Review and refine report
- Prepare presentation slides

**Deliverable:** Technical report draft, presentation slides, code documentation complete

---

#### Hour 18-20: Final Polish & Submission (All)

**Goal:** Final cleanup and submission preparation

**All Together:**

- Code cleanup
  ```bash
  # Format code
  black src/ tests/
  # Lint
  flake8 src/ tests/
  # Security check
  bandit -r src/
  ```
- Final testing
  ```bash
  pytest tests/ -v --cov=src --cov-report=html
  ```
- Repository organization
  - Ensure all files are committed
  - Update README if needed
  - Verify all deliverables
- Create submission package
  - Technical report (PDF)
  - Presentation (PDF/PPT)
  - Code repository (GitHub link)
  - Demo video/instructions

**Deliverable:** Project complete, ready for submission

**Final Checkpoint (Hour 20):** All deliverables complete, project submission-ready

---

### EMR Cluster Usage Windows

| Session       | Hours | Duration     | Purpose                                                    | Owner       | Cost Estimate |
| ------------- | ----- | ------------ | ---------------------------------------------------------- | ----------- | ------------- |
| **Session 1** | 2-4   | 2 hours      | Data Profiling                                             | Hussain     | ~$0.60        |
| **Session 2** | 4-10  | 6 hours      | Feature Engineering, Model Training, Hyperparameter Tuning | Rohit, Ansh | ~$1.80        |
| **Session 3** | 10-16 | 6 hours      | Adversarial Testing, Streaming, Final Integration          | All         | ~$1.80        |
| **Total**     |       | **14 hours** |                                                            |             | **~$4.20**    |

**Cost Optimization:**

- Use m5.large instances (cheaper than m5.xlarge)
- Enable auto-termination
- Terminate immediately after each session
- Use Spot Instances for core nodes (save 70% → ~$1.30 total)

---

### Parallel Work Matrix

| Time Window | Rohit           | Ansh               | Hussain            | Can Work in Parallel?                 |
| ----------- | --------------- | ------------------ | ------------------ | ------------------------------------- |
| H0-1        | Setup           | Setup              | Setup              | ❌ (All together)                     |
| H1-2        | Shared Utils    | Shared Utils       | Shared Utils       | ✅ (Different files)                  |
| H2-4        | Local Dev       | Local Dev          | EMR Profiling      | ✅ (Hussain on EMR, others local)     |
| H4-6        | EMR Features    | Local Dev          | Local Dev          | ✅ (Rohit on EMR, others local)       |
| H6-8        | EMR Training    | EMR Evaluation     | Local Testing      | ✅ (Rohit+Ansh on EMR, Hussain local) |
| H8-10       | Local Prep      | EMR Tuning         | Local Testing      | ✅ (Ansh on EMR, others local)        |
| H10-12      | EMR Setup       | EMR Adversarial    | EMR Adversarial    | ✅ (All on EMR together)              |
| H12-14      | EMR Streaming   | EMR Streaming Eval | Local Testing      | ✅ (Rohit+Ansh on EMR, Hussain local) |
| H14-16      | EMR Integration | EMR Integration    | EMR Integration    | ❌ (All together)                     |
| H16-18      | Documentation   | Documentation      | Documentation Lead | ✅ (All contribute)                   |
| H18-20      | Final Polish    | Final Polish       | Final Polish       | ❌ (All together)                     |

**Key Parallel Opportunities:**

- Hour 2-4: Hussain on EMR, Rohit & Ansh prepare code locally
- Hour 4-6: Rohit on EMR, Ansh & Hussain work locally
- Hour 6-8: Rohit & Ansh on EMR together, Hussain tests locally
- Hour 16-18: All contribute to documentation in parallel

---

### Integration Checkpoints

| Checkpoint       | Hour | What Happens               | Validation                                      |
| ---------------- | ---- | -------------------------- | ----------------------------------------------- |
| **Checkpoint 1** | 2    | Shared utilities merged    | `git log --oneline` shows all common files      |
| **Checkpoint 2** | 4    | Profiling complete         | `aws s3 ls s3://bucket/profiling/` shows report |
| **Checkpoint 3** | 8    | Models trained & evaluated | `aws s3 ls s3://bucket/models/` shows 3 models  |
| **Checkpoint 4** | 10   | Day 1 complete             | All tests pass, models optimized                |
| **Checkpoint 5** | 16   | Advanced features complete | Streaming operational, adversarial tested       |
| **Checkpoint 6** | 20   | Final integration complete | All deliverables ready                          |

**Checkpoint Validation Commands:**

```bash
# Checkpoint 1: Shared utilities
git checkout main
git pull origin main
ls -la src/common/

# Checkpoint 2: Profiling
aws s3 ls s3://$BUCKET_NAME/profiling/ --recursive

# Checkpoint 3: Models
aws s3 ls s3://$BUCKET_NAME/models/ --recursive

# Checkpoint 4: Day 1 complete
pytest tests/ -v
aws s3 ls s3://$BUCKET_NAME/outputs/evaluation/

# Checkpoint 5: Advanced features
aws s3 ls s3://$BUCKET_NAME/outputs/adversarial/
aws s3 ls s3://$BUCKET_NAME/checkpoints/streaming/

# Checkpoint 6: Final
python scripts/final_integration_test.py
```

---

### Cursor-Assisted Development Tips

**Where Cursor Can Accelerate Development:**

1. **Hour 1-2 (Shared Utilities):** Use Cursor to generate boilerplate code for Spark sessions, S3 utils, config files
2. **Hour 2-4 (Code Prep):** Use Cursor to implement MLlib pipelines, evaluation metrics with proper imports
3. **Hour 4-6 (Feature Engineering):** Use Cursor to create VectorAssembler, StandardScaler configurations
4. **Hour 6-8 (Model Training):** Use Cursor to generate model training code for LR, RF, GBT
5. **Hour 8-10 (Hyperparameter Tuning):** Use Cursor to create ParamGridBuilder configurations
6. **Hour 10-12 (Adversarial):** Use Cursor to implement FGSM attack and defense mechanisms
7. **Hour 12-14 (Streaming):** Use Cursor to create Structured Streaming queries
8. **Hour 16-18 (Documentation):** Use Cursor to generate documentation strings and report sections

**Cursor Commands to Use:**

- "Generate PySpark code for [task]"
- "Create test cases for [function]"
- "Add error handling to [code]"
- "Generate documentation for [module]"

---

## Pre-Day 1: Dataset Preparation

**Timeline:** Before Day 1 starts (can be done in advance)  
**Owner:** Rohit (with Hussain validation)

### Step 1: Dataset Validation

Since we're using `creditcard_2023.csv` (already available locally), validate it first:

```bash
# Quick validation (run locally)
python -c "
import pandas as pd
df = pd.read_csv('creditcard_2023.csv/creditcard_2023.csv', nrows=1000)
print(f'Columns: {df.columns.tolist()}')
print(f'Shape sample: {df.shape}')
print(f'Missing values: {df.isnull().sum().sum()}')
if 'Class' in df.columns:
    print(f'Fraud rate: {df[\"Class\"].mean():.4%}')
"
```

**Expected Schema:**

- Time, V1-V28 (features), Amount, Class (target: 0=normal, 1=fraud)

### Step 2: Upload to S3

```bash
# Configure AWS CLI (if not done)
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# Create S3 bucket (replace YOUR_BUCKET_NAME with unique name)
export BUCKET_NAME="csp554-fraud-detection-$(date +%s)"
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Upload dataset (using creditcard_2023.csv)
aws s3 cp creditcard_2023.csv/creditcard_2023.csv s3://$BUCKET_NAME/raw-data/creditcard_2023.csv

# Verify upload
aws s3 ls s3://$BUCKET_NAME/raw-data/
```

### Step 3: Schema Validation

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
│   └── creditcard_2023.csv               # Original dataset
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

### Task Dependencies (2-Day Timeline)

| Task                   | Depends On            | Blocks                 | Owner   | Hour Range | EMR Session |
| ---------------------- | --------------------- | ---------------------- | ------- | ---------- | ----------- |
| Dataset Upload         | Pre-Day 1             | All downstream tasks   | Rohit   | H0-1       | None        |
| Shared Utilities       | Dataset Upload        | Data Profiling         | All     | H1-2       | None        |
| Data Profiling         | Shared Utilities      | Feature Engineering    | Hussain | H2-4       | Session 1   |
| Feature Engineering    | Data Profiling        | Model Training         | Rohit   | H4-6       | Session 2   |
| Model Training         | Feature Engineering   | Model Evaluation       | Rohit   | H6-8       | Session 2   |
| Model Evaluation       | Model Training        | Hyperparameter Tuning  | Ansh    | H6-8       | Session 2   |
| Hyperparameter Tuning  | Model Evaluation      | Adversarial Testing    | Ansh    | H8-10      | Session 2   |
| Adversarial Testing    | Hyperparameter Tuning | Defense Implementation | Ansh    | H10-12     | Session 3   |
| Defense Implementation | Adversarial Testing   | Streaming Integration  | Hussain | H10-12     | Session 3   |
| Streaming Integration  | Model Training        | Streaming Evaluation   | Rohit   | H12-14     | Session 3   |
| Streaming Evaluation   | Streaming Integration | Final Integration      | Ansh    | H12-14     | Session 3   |
| Final Integration      | All components        | Documentation          | All     | H14-16     | Session 3   |
| Documentation          | All components        | Submission             | Hussain | H16-18     | None        |
| Final Polish           | Documentation         | Submission             | All     | H18-20     | None        |

### Critical Path (2-Day Sprint)

```
Hour 0-1: Dataset Upload (Rohit)
    ↓
Hour 1-2: Shared Utilities (All - parallel)
    ↓
Hour 2-4: Data Profiling (Hussain) [EMR Session 1]
    ↓
Hour 4-6: Feature Engineering (Rohit) [EMR Session 2]
    ↓
Hour 6-8: Model Training (Rohit) + Evaluation (Ansh) [EMR Session 2]
    ↓
Hour 8-10: Hyperparameter Tuning (Ansh) [EMR Session 2]
    ↓
Hour 10-12: Adversarial Testing (Ansh + Hussain) [EMR Session 3]
    ↓
Hour 12-14: Streaming Integration (Rohit + Ansh) [EMR Session 3]
    ↓
Hour 14-16: Final Integration (All) [EMR Session 3]
    ↓
Hour 16-18: Documentation (Hussain lead, all contribute)
    ↓
Hour 18-20: Final Polish (All)
```

**Estimated Critical Path Duration:** 20 hours (2 days with parallel work)

### Parallel Work Opportunities

**Can Run in Parallel:**

- Hour 2-4: Hussain (EMR profiling) + Rohit & Ansh (local code prep)
- Hour 4-6: Rohit (EMR features) + Ansh & Hussain (local development)
- Hour 6-8: Rohit & Ansh (EMR training/eval) + Hussain (local testing)
- Hour 8-10: Ansh (EMR tuning) + Rohit & Hussain (local prep/testing)
- Hour 12-14: Rohit & Ansh (EMR streaming) + Hussain (local testing)
- Hour 16-18: All (documentation in parallel)

**Must Run Sequentially:**

- Data Profiling → Feature Engineering (Hussain blocks Rohit)
- Feature Engineering → Model Training (Rohit blocks himself)
- Model Training → Model Evaluation (Rohit blocks Ansh)
- Model Evaluation → Adversarial Testing (Ansh blocks himself)

---

## Cost Management

### AWS Cost Estimates (2-Day Sprint)

| Session                             | Component                 | Instance Type | Duration     | Cost/Hour    | Total Cost |
| ----------------------------------- | ------------------------- | ------------- | ------------ | ------------ | ---------- |
| Pre-Day 1                           | S3 Storage (dataset)      | N/A           | Ongoing      | $0.003/month | $0.003     |
| Session 1                           | EMR Cluster (3x m5.large) | m5.large      | 2 hours      | $0.096/node  | $0.58      |
| Session 2                           | EMR Cluster (3x m5.large) | m5.large      | 6 hours      | $0.096/node  | $1.73      |
| Session 3                           | EMR Cluster (3x m5.large) | m5.large      | 6 hours      | $0.096/node  | $1.73      |
| **Total (On-Demand)**               |                           |               | **14 hours** |              | **~$4.04** |
| **Total (With Spot - 70% savings)** |                           |               | **14 hours** |              | **~$1.21** |

**Cost Breakdown:**

- m5.large: $0.096/hour per node
- 3 nodes (1 master + 2 core): $0.288/hour total
- Session 1 (2h): $0.58
- Session 2 (6h): $1.73
- Session 3 (6h): $1.73
- **Total: ~$4.04** (or **~$1.21 with Spot instances**)

_Note: Costs vary by region and usage. Monitor actual costs in AWS Cost Explorer. Use Spot instances for core nodes to save 70%._

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
  "BudgetName": "fraud-detection-2day-sprint-budget",
  "BudgetLimit": {
    "Amount": "10",
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
