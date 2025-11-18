# Ansh Kaushik - Evaluation & Research Lead

## CSP 554 Big Data Technologies - Fraud Detection Project

**Role:** Evaluation & Research Lead  
**Timeline:** November 18 - December 10, 2025  
**Primary Focus:** Model evaluation, adversarial robustness, streaming evaluation, visualizations

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

1. **Model Evaluation Framework**

   - Implement comprehensive evaluation metrics (AUROC, AUPRC, Precision, Recall, F-score)
   - Design cross-validation strategies for imbalanced data
   - Compare model performance across different algorithms

2. **Hyperparameter Tuning**

   - Set up automated hyperparameter search using Spark MLlib
   - Optimize models for fraud detection (AUROC > 0.90 target)
   - Log experiments to MLflow for reproducibility

3. **Adversarial Robustness Research**

   - Implement FGSM (Fast Gradient Sign Method) for tabular data
   - Evaluate model robustness under adversarial attacks
   - Test defense mechanisms (ensemble voting, feature clamping)

4. **Streaming Evaluation**

   - Design evaluation framework for structured streaming
   - Measure latency and throughput per micro-batch
   - Compare batch vs. streaming performance

5. **Visualization & Reporting**
   - Generate ROC curves, PR curves, confusion matrices
   - Create robustness analysis visualizations
   - Compile results for technical report

### Success Metrics

- ✅ AUROC > 0.90 for all baseline models
- ✅ Complete adversarial robustness evaluation
- ✅ Streaming evaluation framework operational
- ✅ All visualizations generated for report
- ✅ Results reproducible with fixed seeds

### Integration with Team

- **Receives from Rohit:** Trained models, feature-engineered data
- **Receives from Hussain:** Data quality reports, test fixtures
- **Provides to Hussain:** Evaluation metrics, model performance data
- **Provides to Rohit:** Hyperparameter recommendations, performance feedback

---

## Phase-by-Phase Execution Plan

### Phase 1: Evaluation Framework (Nov 18-22)

**Goal:** Build comprehensive evaluation framework with all metrics

#### Day 1 (Nov 18): Setup & Basic Metrics

**Tasks:**

1. Set up Git branch and workspace
2. Install dependencies
3. Implement basic metrics (Accuracy, Precision, Recall)

**Git Commands:**

```bash
# Clone and set up branch
git clone https://github.com/YOUR_USERNAME/BigData_FinalProject.git
cd BigData_FinalProject
git checkout -b feature/ansh-evaluation
git checkout -b feature/ansh-metrics  # Sub-branch for metrics work
```

**File Creation:**

```bash
# Create directory structure
mkdir -p src/ansh/evaluation
mkdir -p src/ansh/visualization
mkdir -p src/ansh/adversarial
mkdir -p src/ansh/streaming
touch src/ansh/__init__.py
touch src/ansh/evaluation/__init__.py
touch src/ansh/visualization/__init__.py
touch src/ansh/adversarial/__init__.py
touch src/ansh/streaming/__init__.py
```

#### Day 2-3 (Nov 19-20): Advanced Metrics Implementation

**Focus:** AUROC, AUPRC, F-score, handling imbalanced data

#### Day 4-5 (Nov 21-22): Cross-Validation & Train-Test Split

**Focus:** Stratified splits, time-based splits, validation framework

**Deliverable:** Complete evaluation framework ready for model testing

---

### Phase 2: Hyperparameter Tuning & Model Comparison (Nov 23-27)

**Goal:** Optimize models and compare performance

#### Day 1-2 (Nov 23-24): Hyperparameter Tuning Setup

**Focus:** CrossValidator, ParamGridBuilder, MLflow integration

#### Day 3-4 (Nov 25-26): Model Comparison

**Focus:** Statistical significance testing, comprehensive comparison

#### Day 5 (Nov 27): Visualization Generation

**Focus:** ROC curves, PR curves, confusion matrices

**Deliverable:** Optimized models with performance visualizations

---

### Phase 3: Adversarial Robustness (Nov 28-Dec 3)

**Goal:** Implement FGSM attacks and evaluate robustness

#### Day 1-2 (Nov 28-29): FGSM Implementation

**Focus:** FGSM for tabular data, gradient computation in Spark

#### Day 3-4 (Nov 30-Dec 1): Attack Evaluation

**Focus:** Attack success rate, robustness metrics

#### Day 5-6 (Dec 2-3): Defense Evaluation

**Focus:** Ensemble voting, feature clamping, robustness comparison

**Deliverable:** Complete adversarial robustness analysis

---

### Phase 4: Streaming Evaluation (Dec 4-6)

**Goal:** Evaluate models in streaming context

#### Day 1-2 (Dec 4-5): Streaming Framework

**Focus:** Structured Streaming integration, latency tracking

#### Day 3 (Dec 6): Performance Comparison

**Focus:** Batch vs. streaming accuracy, throughput analysis

**Deliverable:** Streaming evaluation results

---

### Phase 5: Results Compilation & Demo (Dec 7-10)

**Goal:** Finalize all results, prepare demo and presentation

#### Day 1-2 (Dec 7-8): Result Compilation

**Focus:** Aggregate metrics, generate final visualizations

#### Day 3-4 (Dec 9-10): Demo & Presentation

**Focus:** Demo scenario preparation, presentation slides

**Deliverable:** Complete evaluation package ready for submission

---

## File Structure

```
src/ansh/
├── __init__.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                    # All evaluation metrics
│   ├── cross_validation.py           # CV strategies
│   └── model_comparison.py           # Model comparison framework
├── adversarial/
│   ├── __init__.py
│   ├── fgsm_attack.py                # FGSM implementation
│   ├── perturbation_utils.py         # Perturbation helpers
│   └── robustness_metrics.py         # Robustness evaluation
├── streaming/
│   ├── __init__.py
│   ├── streaming_evaluator.py        # Streaming evaluation
│   └── latency_tracker.py            # Latency measurement
└── visualization/
    ├── __init__.py
    ├── plot_roc.py                   # ROC curve plotting
    ├── plot_robustness.py            # Robustness visualizations
    └── generate_report_plots.py       # All report figures
```

---

## Setup & Configuration

### Dependencies Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install core dependencies
pip install pyspark==3.5.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0

# Install MLflow for experiment tracking
pip install mlflow==2.8.1

# Install additional utilities
pip install boto3==1.28.25
pip install pyarrow==13.0.0

# Save to requirements.txt
pip freeze > requirements.txt
```

### AWS CLI Configuration

```bash
# Configure AWS CLI (if not done)
aws configure
# Enter:
# - AWS Access Key ID: [Your key]
# - AWS Secret Access Key: [Your secret]
# - Default region: us-east-1
# - Default output format: json

# Verify configuration
aws sts get-caller-identity

# Set environment variables
export S3_BUCKET="csp554-fraud-detection-XXXXX"  # Replace with actual bucket
export AWS_REGION="us-east-1"
```

### Spark Session Configuration

**File:** `src/ansh/evaluation/spark_config.py`

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from src.common.config import config

def create_evaluation_spark_session():
    """Create Spark session optimized for evaluation tasks."""
    spark = SparkSession.builder \
        .appName("FraudDetectionEvaluation") \
        .master("yarn") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark
```

### MLflow Setup

**File:** `src/ansh/evaluation/mlflow_setup.py`

```python
import mlflow
import mlflow.spark
from src.common.config import config

def setup_mlflow_tracking():
    """Set up MLflow tracking for experiments."""
    # Option 1: Local tracking (for development)
    # mlflow.set_tracking_uri("file:./mlruns")

    # Option 2: S3-backed tracking (for EMR)
    mlflow.set_tracking_uri(f"s3://{config.S3_BUCKET}/mlflow/")

    # Create experiment
    experiment_name = "fraud_detection_evaluation"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id

# Usage
if __name__ == "__main__":
    setup_mlflow_tracking()
    print("MLflow tracking configured")
```

---

## Code Implementations

### Phase 1: Evaluation Metrics

**File:** `src/ansh/evaluation/metrics.py`

```python
"""
Comprehensive evaluation metrics for fraud detection models.
Handles imbalanced data and provides multiple performance measures.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationMetrics
import numpy as np
from typing import Dict, List, Tuple
import json

class FraudDetectionMetrics:
    """Comprehensive metrics calculator for fraud detection."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        self.multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )

    def calculate_all_metrics(self, predictions_df) -> Dict[str, float]:
        """
        Calculate all evaluation metrics for fraud detection.

        Args:
            predictions_df: DataFrame with columns: label, prediction, probability

        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        accuracy = self._calculate_accuracy(predictions_df)
        precision = self._calculate_precision(predictions_df)
        recall = self._calculate_recall(predictions_df)
        f1_score = self._calculate_f1_score(predictions_df)

        # Advanced metrics
        auroc = self._calculate_auroc(predictions_df)
        auprc = self._calculate_auprc(predictions_df)

        # Confusion matrix
        confusion_matrix = self._calculate_confusion_matrix(predictions_df)

        # Imbalanced data metrics
        specificity = self._calculate_specificity(confusion_matrix)
        balanced_accuracy = (recall + specificity) / 2.0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auroc": auroc,
            "auprc": auprc,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "confusion_matrix": confusion_matrix,
            "true_positives": int(confusion_matrix["tp"]),
            "true_negatives": int(confusion_matrix["tn"]),
            "false_positives": int(confusion_matrix["fp"]),
            "false_negatives": int(confusion_matrix["fn"])
        }

        return metrics

    def _calculate_accuracy(self, predictions_df) -> float:
        """Calculate accuracy."""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_precision(self, predictions_df) -> float:
        """Calculate precision (positive predictive value)."""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_recall(self, predictions_df) -> float:
        """Calculate recall (sensitivity, true positive rate)."""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_f1_score(self, predictions_df) -> float:
        """Calculate F1 score."""
        return self.multiclass_evaluator.evaluate(predictions_df)

    def _calculate_auroc(self, predictions_df) -> float:
        """Calculate Area Under ROC Curve."""
        return self.binary_evaluator.evaluate(predictions_df)

    def _calculate_auprc(self, predictions_df) -> float:
        """Calculate Area Under Precision-Recall Curve."""
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_confusion_matrix(self, predictions_df) -> Dict[str, int]:
        """Calculate confusion matrix components."""
        # Convert to Pandas for easier calculation (small dataset)
        # For larger datasets, use Spark aggregations
        pred_pandas = predictions_df.select("label", "prediction").toPandas()

        tp = len(pred_pandas[(pred_pandas["label"] == 1) & (pred_pandas["prediction"] == 1)])
        tn = len(pred_pandas[(pred_pandas["label"] == 0) & (pred_pandas["prediction"] == 0)])
        fp = len(pred_pandas[(pred_pandas["label"] == 0) & (pred_pandas["prediction"] == 1)])
        fn = len(pred_pandas[(pred_pandas["label"] == 1) & (pred_pandas["prediction"] == 0)])

        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def _calculate_specificity(self, confusion_matrix: Dict[str, int]) -> float:
        """Calculate specificity (true negative rate)."""
        tn = confusion_matrix["tn"]
        fp = confusion_matrix["fp"]
        if (tn + fp) == 0:
            return 0.0
        return tn / (tn + fp)

    def save_metrics_to_s3(self, metrics: Dict, model_name: str, version: str = "v1"):
        """Save metrics to S3 as JSON."""
        from src.common.config import config
        import boto3
        import json

        s3_client = boto3.client('s3')
        metrics_json = json.dumps(metrics, indent=2)

        s3_key = f"outputs/evaluation/{model_name}_{version}_metrics.json"
        s3_client.put_object(
            Bucket=config.S3_BUCKET,
            Key=s3_key,
            Body=metrics_json,
            ContentType="application/json"
        )
        print(f"Metrics saved to s3://{config.S3_BUCKET}/{s3_key}")

    @staticmethod
    def load_metrics_from_s3(s3_path: str) -> Dict:
        """Load metrics from S3."""
        import boto3
        import json

        s3_client = boto3.client('s3')
        bucket, key = s3_path.replace("s3://", "").split("/", 1)

        response = s3_client.get_object(Bucket=bucket, Key=key)
        metrics = json.loads(response['Body'].read().decode('utf-8'))
        return metrics

# Usage example
if __name__ == "__main__":
    from src.ansh.evaluation.spark_config import create_evaluation_spark_session
    from pyspark.ml import PipelineModel

    spark = create_evaluation_spark_session()

    # Load model and test data (from Rohit's pipeline)
    model = PipelineModel.load("s3://bucket/models/logistic_regression/v1/model/")
    test_df = spark.read.parquet("s3://bucket/processed/test/")

    # Make predictions
    predictions = model.transform(test_df)

    # Calculate metrics
    metrics_calculator = FraudDetectionMetrics(spark)
    metrics = metrics_calculator.calculate_all_metrics(predictions)

    # Print results
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")

    # Save to S3
    metrics_calculator.save_metrics_to_s3(metrics, "logistic_regression", "v1")

    spark.stop()
```

**File:** `src/ansh/evaluation/cross_validation.py`

```python
"""
Cross-validation strategies for imbalanced fraud detection data.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from typing import List, Dict
import numpy as np

class ImbalancedDataSplitter:
    """Handle train-test splits for imbalanced data."""

    def __init__(self, spark: SparkSession, random_seed: int = 42):
        self.spark = spark
        self.random_seed = random_seed

    def stratified_train_test_split(
        self,
        df: DataFrame,
        train_ratio: float = 0.8,
        label_col: str = "Class"
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Stratified train-test split maintaining fraud ratio.

        Args:
            df: Input DataFrame
            train_ratio: Proportion for training (default 0.8)
            label_col: Name of label column

        Returns:
            Tuple of (train_df, test_df)
        """
        # Calculate fraud rate
        total_count = df.count()
        fraud_count = df.filter(F.col(label_col) == 1).count()
        fraud_rate = fraud_count / total_count

        # Split fraud and non-fraud separately
        fraud_df = df.filter(F.col(label_col) == 1)
        non_fraud_df = df.filter(F.col(label_col) == 0)

        # Split each class
        fraud_train, fraud_test = fraud_df.randomSplit(
            [train_ratio, 1 - train_ratio],
            seed=self.random_seed
        )
        non_fraud_train, non_fraud_test = non_fraud_df.randomSplit(
            [train_ratio, 1 - train_ratio],
            seed=self.random_seed
        )

        # Combine
        train_df = fraud_train.union(non_fraud_train)
        test_df = fraud_test.union(non_fraud_test)

        # Verify fraud rate maintained
        train_fraud_rate = train_df.filter(F.col(label_col) == 1).count() / train_df.count()
        test_fraud_rate = test_df.filter(F.col(label_col) == 1).count() / test_df.count()

        print(f"Original fraud rate: {fraud_rate:.4f}")
        print(f"Train fraud rate: {train_fraud_rate:.4f}")
        print(f"Test fraud rate: {test_fraud_rate:.4f}")

        return train_df, test_df

    def time_based_split(
        self,
        df: DataFrame,
        time_col: str = "Time",
        split_time: float = None,
        train_ratio: float = 0.8
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Time-based split (for temporal data).

        Args:
            df: Input DataFrame
            time_col: Name of time column
            split_time: Specific time to split (if None, uses train_ratio)
            train_ratio: Proportion for training

        Returns:
            Tuple of (train_df, test_df)
        """
        if split_time is None:
            # Calculate split time based on ratio
            max_time = df.agg(F.max(time_col).alias("max_time")).collect()[0]["max_time"]
            split_time = max_time * train_ratio

        train_df = df.filter(F.col(time_col) <= split_time)
        test_df = df.filter(F.col(time_col) > split_time)

        print(f"Split time: {split_time}")
        print(f"Train size: {train_df.count():,}")
        print(f"Test size: {test_df.count():,}")

        return train_df, test_df

class HyperparameterTuner:
    """Hyperparameter tuning with cross-validation."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def tune_model(
        self,
        pipeline: Pipeline,
        train_df: DataFrame,
        param_grid: Dict,
        num_folds: int = 5,
        metric_name: str = "areaUnderROC"
    ):
        """
        Perform hyperparameter tuning using cross-validation.

        Args:
            pipeline: MLlib Pipeline
            train_df: Training DataFrame
            param_grid: Dictionary of parameter grids
            num_folds: Number of CV folds
            metric_name: Metric to optimize

        Returns:
            Best model and best parameters
        """
        # Create evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName=metric_name
        )

        # Build parameter grid
        param_grid_builder = ParamGridBuilder()
        for param_name, values in param_grid.items():
            param_grid_builder = param_grid_builder.addGrid(
                getattr(pipeline.getStages()[-1], param_name),
                values
            )
        param_grid = param_grid_builder.build()

        # Create cross-validator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42,
            parallelism=4  # Number of parallel jobs
        )

        # Fit model
        print("Starting cross-validation...")
        cv_model = cv.fit(train_df)

        # Get best model
        best_model = cv_model.bestModel
        best_params = cv_model.bestModel.extractParamMap()

        # Get CV scores
        avg_metrics = cv_model.avgMetrics
        best_score = max(avg_metrics)

        print(f"\n=== Cross-Validation Results ===")
        print(f"Best {metric_name}: {best_score:.4f}")
        print(f"Best parameters:")
        for param, value in best_params.items():
            print(f"  {param.name}: {value}")

        return best_model, best_params, avg_metrics

# Usage example
if __name__ == "__main__":
    from src.ansh.evaluation.spark_config import create_evaluation_spark_session
    from src.common.config import config

    spark = create_evaluation_spark_session()

    # Load data
    df = spark.read.csv(config.S3_RAW_DATA_PATH, header=True, inferSchema=True)

    # Stratified split
    splitter = ImbalancedDataSplitter(spark, random_seed=config.RANDOM_SEED)
    train_df, test_df = splitter.stratified_train_test_split(df, train_ratio=0.8)

    # Save splits
    train_df.write.mode("overwrite").parquet(f"{config.S3_PROCESSED_PATH}/train/")
    test_df.write.mode("overwrite").parquet(f"{config.S3_PROCESSED_PATH}/test/")

    spark.stop()
```

**File:** `src/ansh/evaluation/model_comparison.py`

```python
"""
Model comparison framework with statistical significance testing.
"""
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from src.ansh.evaluation.metrics import FraudDetectionMetrics
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy import stats

class ModelComparator:
    """Compare multiple models with statistical testing."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.metrics_calculator = FraudDetectionMetrics(spark)
        self.results = []

    def compare_models(
        self,
        models: Dict[str, PipelineModel],
        test_df,
        model_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on test data.

        Args:
            models: Dictionary of {model_name: PipelineModel}
            test_df: Test DataFrame
            model_names: Optional list of model names (uses dict keys if None)

        Returns:
            DataFrame with comparison results
        """
        if model_names is None:
            model_names = list(models.keys())

        comparison_results = []

        for model_name in model_names:
            model = models[model_name]

            # Make predictions
            predictions = model.transform(test_df)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(predictions)
            metrics["model_name"] = model_name

            comparison_results.append(metrics)
            self.results.append(metrics)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)

        # Sort by AUROC (descending)
        comparison_df = comparison_df.sort_values("auroc", ascending=False)

        return comparison_df

    def statistical_significance_test(
        self,
        model1_predictions,
        model2_predictions,
        metric: str = "auroc"
    ) -> Dict:
        """
        Perform statistical significance test between two models.

        Args:
            model1_predictions: Predictions from model 1
            model2_predictions: Predictions from model 2
            metric: Metric to compare

        Returns:
            Dictionary with test results
        """
        # Calculate metrics for each fold (if using CV) or bootstrap
        # For simplicity, using bootstrap sampling

        n_bootstrap = 1000
        model1_scores = []
        model2_scores = []

        # Bootstrap sampling
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample1 = model1_predictions.sample(True, 1.0, seed=np.random.randint(0, 10000))
            sample2 = model2_predictions.sample(True, 1.0, seed=np.random.randint(0, 10000))

            metrics1 = self.metrics_calculator.calculate_all_metrics(sample1)
            metrics2 = self.metrics_calculator.calculate_all_metrics(sample2)

            model1_scores.append(metrics1[metric])
            model2_scores.append(metrics2[metric])

        # Perform paired t-test
        differences = np.array(model1_scores) - np.array(model2_scores)
        t_stat, p_value = stats.ttest_1samp(differences, 0)

        result = {
            "model1_mean": np.mean(model1_scores),
            "model2_mean": np.mean(model2_scores),
            "difference": np.mean(differences),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

        return result

    def generate_comparison_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate formatted comparison report."""
        report = "\n=== Model Comparison Report ===\n\n"
        report += comparison_df[["model_name", "auroc", "auprc", "precision", "recall", "f1_score"]].to_string(index=False)
        report += "\n\n"

        # Best model
        best_model = comparison_df.iloc[0]
        report += f"Best Model: {best_model['model_name']}\n"
        report += f"  AUROC: {best_model['auroc']:.4f}\n"
        report += f"  AUPRC: {best_model['auprc']:.4f}\n"
        report += f"  F1 Score: {best_model['f1_score']:.4f}\n"

        return report

# Usage example
if __name__ == "__main__":
    from src.ansh.evaluation.spark_config import create_evaluation_spark_session
    from src.common.config import config

    spark = create_evaluation_spark_session()

    # Load models
    models = {
        "logistic_regression": PipelineModel.load(f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/"),
        "random_forest": PipelineModel.load(f"{config.S3_MODELS_PATH}/random_forest/v1/model/"),
        "gbt_classifier": PipelineModel.load(f"{config.S3_MODELS_PATH}/gbt_classifier/v1/model/")
    }

    # Load test data
    test_df = spark.read.parquet(f"{config.S3_PROCESSED_PATH}/test/")

    # Compare models
    comparator = ModelComparator(spark)
    comparison_df = comparator.compare_models(models, test_df)

    # Print report
    print(comparator.generate_comparison_report(comparison_df))

    # Save comparison
    comparison_df.to_csv("model_comparison.csv", index=False)

    spark.stop()
```

### Phase 2: Hyperparameter Tuning with MLflow

**File:** `src/ansh/evaluation/hyperparameter_tuning.py`

```python
"""
Hyperparameter tuning with MLflow tracking.
"""
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from src.ansh.evaluation.cross_validation import HyperparameterTuner
from src.ansh.evaluation.metrics import FraudDetectionMetrics
from src.ansh.evaluation.mlflow_setup import setup_mlflow_tracking

def tune_logistic_regression(spark: SparkSession, train_df, test_df):
    """Tune Logistic Regression hyperparameters."""
    setup_mlflow_tracking()

    # Create pipeline
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="Class")

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # Define parameter grid
    param_grid = {
        "regParam": [0.01, 0.1, 1.0, 10.0],
        "elasticNetParam": [0.0, 0.5, 1.0],
        "maxIter": [100, 200, 300]
    }

    # Tune
    tuner = HyperparameterTuner(spark)

    with mlflow.start_run(run_name="LR_Hyperparameter_Tuning"):
        best_model, best_params, cv_scores = tuner.tune_model(
            pipeline, train_df, param_grid, num_folds=5
        )

        # Evaluate on test set
        test_predictions = best_model.transform(test_df)
        metrics_calc = FraudDetectionMetrics(spark)
        test_metrics = metrics_calc.calculate_all_metrics(test_predictions)

        # Log to MLflow
        mlflow.log_params({str(k.name): v for k, v in best_params.items()})
        mlflow.log_metrics({
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1_score"],
            "cv_mean_auroc": np.mean(cv_scores),
            "cv_std_auroc": np.std(cv_scores)
        })

        # Log model
        mlflow.spark.log_model(best_model, "model")

        print(f"Best test AUROC: {test_metrics['auroc']:.4f}")

    return best_model, test_metrics

# Similar functions for RandomForest and GBT...
# (See full implementation in repository)

if __name__ == "__main__":
    from src.ansh.evaluation.spark_config import create_evaluation_spark_session
    from src.common.config import config

    spark = create_evaluation_spark_session()

    train_df = spark.read.parquet(f"{config.S3_PROCESSED_PATH}/train/")
    test_df = spark.read.parquet(f"{config.S3_PROCESSED_PATH}/test/")

    # Tune models
    lr_model, lr_metrics = tune_logistic_regression(spark, train_df, test_df)

    spark.stop()
```

### Phase 3: Adversarial Robustness

**File:** `src/ansh/adversarial/fgsm_attack.py`

```python
"""
Fast Gradient Sign Method (FGSM) implementation for tabular fraud detection.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
import numpy as np
from typing import List, Tuple
from pyspark.sql.types import ArrayType, DoubleType

class FGSMAttack:
    """FGSM attack implementation for Spark MLlib models."""

    def __init__(self, spark: SparkSession, model: PipelineModel, epsilon: float = 0.1):
        """
        Initialize FGSM attack.

        Args:
            spark: Spark session
            model: Trained PipelineModel
            epsilon: Perturbation magnitude
        """
        self.spark = spark
        self.model = model
        self.epsilon = epsilon

    def generate_adversarial_examples(
        self,
        df: DataFrame,
        feature_cols: List[str],
        label_col: str = "Class"
    ) -> DataFrame:
        """
        Generate adversarial examples using FGSM.

        Args:
            df: Input DataFrame with features
            feature_cols: List of feature column names
            label_col: Name of label column

        Returns:
            DataFrame with adversarial examples
        """
        # Get predictions on original data
        predictions = self.model.transform(df)

        # Calculate gradients (simplified approach for tabular data)
        # For Spark MLlib, we need to approximate gradients
        adversarial_df = self._apply_fgsm_perturbation(
            df, predictions, feature_cols, label_col
        )

        return adversarial_df

    def _apply_fgsm_perturbation(
        self,
        df: DataFrame,
        predictions: DataFrame,
        feature_cols: List[str],
        label_col: str
    ) -> DataFrame:
        """Apply FGSM perturbation to features."""
        # For tabular data, we perturb each feature by epsilon * sign of feature importance
        # This is a simplified FGSM - full gradient computation requires model internals

        adversarial_df = df

        for col in feature_cols:
            # Calculate perturbation: epsilon * sign(feature)
            # In practice, this would use actual gradients
            perturbation = F.when(F.col(col) >= 0, self.epsilon).otherwise(-self.epsilon)
            adversarial_df = adversarial_df.withColumn(
                f"{col}_adversarial",
                F.col(col) + perturbation
            )

        # Select adversarial columns
        adversarial_cols = [f"{col}_adversarial" for col in feature_cols]
        adversarial_df = adversarial_df.select(
            *[F.col(f"{col}_adversarial").alias(col) for col in feature_cols],
            label_col
        )

        return adversarial_df

    def evaluate_attack_success(
        self,
        original_predictions: DataFrame,
        adversarial_predictions: DataFrame,
        label_col: str = "Class"
    ) -> Dict:
        """
        Calculate attack success rate (ASR).

        Args:
            original_predictions: Predictions on clean data
            adversarial_predictions: Predictions on adversarial data
            label_col: Name of label column

        Returns:
            Dictionary with attack metrics
        """
        # Convert to Pandas for easier comparison
        orig_pd = original_predictions.select("prediction", label_col).toPandas()
        adv_pd = adversarial_predictions.select("prediction", label_col).toPandas()

        # Calculate metrics
        total_samples = len(orig_pd)
        correct_original = (orig_pd["prediction"] == orig_pd[label_col]).sum()
        correct_adversarial = (adv_pd["prediction"] == adv_pd[label_col]).sum()

        original_accuracy = correct_original / total_samples
        adversarial_accuracy = correct_adversarial / total_samples

        # Attack success: predictions that changed from correct to incorrect
        attack_success = ((orig_pd["prediction"] == orig_pd[label_col]) &
                         (adv_pd["prediction"] != adv_pd[label_col])).sum()
        attack_success_rate = attack_success / total_samples

        metrics = {
            "original_accuracy": original_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "accuracy_drop": original_accuracy - adversarial_accuracy,
            "attack_success_rate": attack_success_rate,
            "total_samples": total_samples
        }

        return metrics

# Usage example
if __name__ == "__main__":
    from src.ansh.evaluation.spark_config import create_evaluation_spark_session
    from src.common.config import config
    from pyspark.ml import PipelineModel

    spark = create_evaluation_spark_session()

    # Load model
    model = PipelineModel.load(f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/")

    # Load test data
    test_df = spark.read.parquet(f"{config.S3_PROCESSED_PATH}/test/")

    # Test different epsilon values
    epsilons = [0.01, 0.05, 0.1, 0.2]
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    results = []
    for epsilon in epsilons:
        print(f"\nTesting epsilon = {epsilon}")
        attack = FGSMAttack(spark, model, epsilon=epsilon)

        # Generate adversarial examples
        adversarial_df = attack.generate_adversarial_examples(test_df, feature_cols)

        # Evaluate
        original_predictions = model.transform(test_df)
        adversarial_predictions = model.transform(adversarial_df)

        metrics = attack.evaluate_attack_success(
            original_predictions, adversarial_predictions
        )
        metrics["epsilon"] = epsilon
        results.append(metrics)

        print(f"Original Accuracy: {metrics['original_accuracy']:.4f}")
        print(f"Adversarial Accuracy: {metrics['adversarial_accuracy']:.4f}")
        print(f"Attack Success Rate: {metrics['attack_success_rate']:.4f}")

    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv("adversarial_robustness_results.csv", index=False)

    spark.stop()
```

**File:** `src/ansh/adversarial/robustness_metrics.py`

```python
"""
Robustness evaluation metrics for adversarial attacks.
"""
from typing import Dict, List
import numpy as np
import pandas as pd

class RobustnessEvaluator:
    """Evaluate model robustness under adversarial attacks."""

    @staticmethod
    def calculate_robustness_score(
        clean_accuracy: float,
        adversarial_accuracy: float,
        epsilon: float
    ) -> float:
        """
        Calculate robustness score.

        Higher score = more robust.
        Score = 1 - (accuracy_drop / epsilon)
        """
        accuracy_drop = clean_accuracy - adversarial_accuracy
        if epsilon == 0:
            return 1.0 if accuracy_drop == 0 else 0.0

        robustness = 1.0 - (accuracy_drop / epsilon)
        return max(0.0, min(1.0, robustness))  # Clamp to [0, 1]

    @staticmethod
    def evaluate_robustness_curve(
        results: List[Dict]
    ) -> pd.DataFrame:
        """
        Evaluate robustness across different epsilon values.

        Args:
            results: List of dictionaries with epsilon and accuracy metrics

        Returns:
            DataFrame with robustness analysis
        """
        df = pd.DataFrame(results)

        # Calculate robustness scores
        df["robustness_score"] = df.apply(
            lambda row: RobustnessEvaluator.calculate_robustness_score(
                row["original_accuracy"],
                row["adversarial_accuracy"],
                row["epsilon"]
            ),
            axis=1
        )

        # Calculate area under robustness curve (AURC)
        df_sorted = df.sort_values("epsilon")
        aurc = np.trapz(df_sorted["robustness_score"], df_sorted["epsilon"])
        df["aurc"] = aurc

        return df

# Usage
if __name__ == "__main__":
    # Load results from FGSM attack
    results_df = pd.read_csv("adversarial_robustness_results.csv")

    evaluator = RobustnessEvaluator()
    robustness_df = evaluator.evaluate_robustness_curve(results_df.to_dict("records"))

    print(robustness_df)
    robustness_df.to_csv("robustness_analysis.csv", index=False)
```

### Phase 4: Streaming Evaluation

**File:** `src/ansh/streaming/streaming_evaluator.py`

```python
"""
Evaluation framework for structured streaming.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from src.ansh.evaluation.metrics import FraudDetectionMetrics
from typing import Dict
import time

class StreamingEvaluator:
    """Evaluate models in streaming context."""

    def __init__(self, spark: SparkSession, model: PipelineModel):
        self.spark = spark
        self.model = model
        self.metrics_calculator = FraudDetectionMetrics(spark)
        self.metrics_history = []

    def evaluate_streaming_batch(
        self,
        batch_df: DataFrame,
        batch_id: int
    ) -> Dict:
        """
        Evaluate a single streaming batch.

        Args:
            batch_df: DataFrame from streaming batch
            batch_id: Batch identifier

        Returns:
            Dictionary with batch metrics
        """
        start_time = time.time()

        # Make predictions
        predictions = self.model.transform(batch_df)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(predictions)

        # Calculate latency
        latency = time.time() - start_time

        # Add batch metadata
        batch_metrics = {
            "batch_id": batch_id,
            "batch_size": batch_df.count(),
            "latency_seconds": latency,
            **metrics
        }

        self.metrics_history.append(batch_metrics)

        return batch_metrics

    def aggregate_streaming_metrics(self) -> Dict:
        """Aggregate metrics across all batches."""
        if not self.metrics_history:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.metrics_history)

        aggregated = {
            "total_batches": len(df),
            "total_samples": df["batch_size"].sum(),
            "avg_latency": df["latency_seconds"].mean(),
            "p95_latency": df["latency_seconds"].quantile(0.95),
            "p99_latency": df["latency_seconds"].quantile(0.99),
            "avg_auroc": df["auroc"].mean(),
            "avg_auprc": df["auprc"].mean(),
            "avg_throughput": df["batch_size"].sum() / df["latency_seconds"].sum()
        }

        return aggregated

# Usage in streaming context
def evaluate_streaming_model():
    """Example streaming evaluation."""
    from src.ansh.evaluation.spark_config import create_evaluation_spark_session
    from src.common.config import config
    from pyspark.ml import PipelineModel

    spark = create_evaluation_spark_session()

    # Load model
    model = PipelineModel.load(f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/")

    # Create streaming evaluator
    evaluator = StreamingEvaluator(spark, model)

    # Read streaming data
    streaming_df = spark.readStream \
        .schema(get_credit_card_schema()) \
        .csv(f"{config.S3_PROCESSED_PATH}/streaming/")

    # Process stream
    def process_batch(batch_df, batch_id):
        metrics = evaluator.evaluate_streaming_batch(batch_df, batch_id)
        print(f"Batch {batch_id}: AUROC = {metrics['auroc']:.4f}, Latency = {metrics['latency_seconds']:.2f}s")

    query = streaming_df.writeStream \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", f"{config.S3_BUCKET}/checkpoints/streaming/") \
        .start()

    query.awaitTermination(timeout=300)  # Run for 5 minutes

    # Get aggregated metrics
    aggregated = evaluator.aggregate_streaming_metrics()
    print("\n=== Streaming Evaluation Results ===")
    print(f"Total Batches: {aggregated['total_batches']}")
    print(f"Average Latency: {aggregated['avg_latency']:.2f}s")
    print(f"Throughput: {aggregated['avg_throughput']:.2f} samples/sec")

    spark.stop()
```

### Phase 5: Visualizations

**File:** `src/ansh/visualization/plot_roc.py`

```python
"""
ROC curve visualization.
"""
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationMetrics
from pyspark.sql import DataFrame
import numpy as np

def plot_roc_curve(
    predictions_df: DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
    probability_col: str = "probability",
    save_path: str = "roc_curve.png"
):
    """
    Plot ROC curve from predictions.

    Args:
        predictions_df: DataFrame with predictions
        label_col: Name of label column
        prediction_col: Name of prediction column
        probability_col: Name of probability column
        save_path: Path to save figure
    """
    # Get RDD for BinaryClassificationMetrics
    prediction_and_labels = predictions_df.select(
        probability_col, label_col
    ).rdd.map(lambda row: (float(row[probability_col][1]), float(row[label_col])))

    # Calculate metrics
    metrics = BinaryClassificationMetrics(prediction_and_labels)

    # Get ROC curve
    fpr = []
    tpr = []
    for curve in metrics.roc().collect():
        fpr.append(curve[0])
        tpr.append(curve[1])

    # Calculate AUC
    auc = metrics.areaUnderROC

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Fraud Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved to {save_path}")

# Similar functions for PR curves, confusion matrices, etc.
# (See full implementation in repository)
```

---

## Integration Points

### Importing Rohit's Models

```python
# In your evaluation scripts
from pyspark.ml import PipelineModel
from src.common.config import config

# Load trained model
model = PipelineModel.load(f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/")

# Use model for evaluation
predictions = model.transform(test_df)
```

### Using Hussain's Profiling Data

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
feature_importance = profiling_metadata.get('feature_importance', {})
```

### Providing Results to Team

```python
# Save metrics for Hussain's testing
from src.ansh.evaluation.metrics import FraudDetectionMetrics

metrics_calc = FraudDetectionMetrics(spark)
metrics = metrics_calc.calculate_all_metrics(predictions)

# Save to S3 (accessible to all)
metrics_calc.save_metrics_to_s3(metrics, "logistic_regression", "v1")
```

---

## Quality Gates

### Pre-Commit Checks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### Unit Tests

**File:** `tests/test_ansh_evaluation.py`

```python
import pytest
from pyspark.sql import SparkSession
from src.ansh.evaluation.metrics import FraudDetectionMetrics

@pytest.fixture
def spark():
    return SparkSession.builder.master("local[2]").appName("test").getOrCreate()

def test_metrics_calculation(spark):
    # Create test data
    # ... (test implementation)
    pass
```

**Run tests:**

```bash
pytest tests/test_ansh_evaluation.py -v
```

### Validation Checks

- [ ] All metrics return values in [0, 1] range
- [ ] AUROC > 0.90 for baseline models
- [ ] No hardcoded paths (use config.py)
- [ ] All functions have docstrings
- [ ] Code passes linting (flake8, black)

---

## Daily Task Breakdown

### Phase 1 Daily Tasks (Nov 18-22)

| Day    | Task                        | Hours | Deliverable                   |
| ------ | --------------------------- | ----- | ----------------------------- |
| Nov 18 | Setup, basic metrics        | 4     | metrics.py skeleton           |
| Nov 19 | AUROC, AUPRC implementation | 6     | Advanced metrics working      |
| Nov 20 | F-score, confusion matrix   | 4     | All metrics implemented       |
| Nov 21 | Stratified split            | 4     | cross_validation.py           |
| Nov 22 | Testing & integration       | 4     | Complete evaluation framework |

### Phase 2 Daily Tasks (Nov 23-27)

| Day    | Task                       | Hours | Deliverable                  |
| ------ | -------------------------- | ----- | ---------------------------- |
| Nov 23 | MLflow setup, CV framework | 6     | Hyperparameter tuning script |
| Nov 24 | Tune Logistic Regression   | 4     | Optimized LR model           |
| Nov 25 | Tune RF and GBT            | 6     | All models optimized         |
| Nov 26 | Model comparison           | 4     | Comparison report            |
| Nov 27 | Visualizations             | 4     | ROC/PR curves generated      |

### Phase 3 Daily Tasks (Nov 28-Dec 3)

| Day     | Task                | Hours | Deliverable                  |
| ------- | ------------------- | ----- | ---------------------------- |
| Nov 28  | FGSM implementation | 6     | fgsm_attack.py               |
| Nov 29  | Test FGSM on models | 4     | Attack results               |
| Nov 30  | Robustness metrics  | 4     | Robustness analysis          |
| Dec 1   | Defense evaluation  | 6     | Defense results              |
| Dec 2-3 | Documentation       | 4     | Adversarial section complete |

### Phase 4 Daily Tasks (Dec 4-6)

| Day   | Task                | Hours | Deliverable            |
| ----- | ------------------- | ----- | ---------------------- |
| Dec 4 | Streaming evaluator | 6     | streaming_evaluator.py |
| Dec 5 | Latency tracking    | 4     | Performance metrics    |
| Dec 6 | Batch vs. streaming | 4     | Comparison results     |

### Phase 5 Daily Tasks (Dec 7-10)

| Day    | Task                 | Hours | Deliverable          |
| ------ | -------------------- | ----- | -------------------- |
| Dec 7  | Result compilation   | 6     | Final metrics report |
| Dec 8  | Visualization polish | 4     | Report-ready figures |
| Dec 9  | Demo preparation     | 4     | Demo script ready    |
| Dec 10 | Presentation         | 4     | Slides complete      |

---

## Checkpoint Questions

### After Phase 1

- [ ] Can you calculate AUROC for a model?
- [ ] Does stratified split maintain fraud rate?
- [ ] Are all metrics returning valid values?

### After Phase 2

- [ ] Are hyperparameters optimized?
- [ ] Is MLflow tracking working?
- [ ] Can you compare models statistically?

### After Phase 3

- [ ] Can you generate adversarial examples?
- [ ] Is attack success rate calculated correctly?
- [ ] Are robustness metrics meaningful?

### After Phase 4

- [ ] Can you evaluate streaming batches?
- [ ] Is latency being tracked?
- [ ] Are batch vs. streaming results comparable?

### After Phase 5

- [ ] Are all visualizations generated?
- [ ] Can you reproduce all results?
- [ ] Is demo ready for presentation?

---

## Pro Tips

1. **Use Caching:** Cache DataFrames used multiple times in evaluation
2. **Parallelize CV:** Set `parallelism` in CrossValidator for faster tuning
3. **MLflow Logging:** Log everything - parameters, metrics, artifacts
4. **Reproducibility:** Always set random seeds (42)
5. **Incremental Testing:** Test each metric function independently
6. **Visualization Early:** Generate plots during development for debugging
7. **S3 Organization:** Use versioned paths for all outputs
8. **Error Handling:** Wrap metric calculations in try-except blocks
9. **Documentation:** Document all assumptions (e.g., epsilon ranges)
10. **Performance:** Use Spark aggregations instead of Pandas when possible

---

**Document Version:** 1.0  
**Last Updated:** November 18, 2025  
**Owner:** Ansh Kaushik
