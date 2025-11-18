# Hussain Bin Yousuf - Quality Assurance & Documentation Lead

## CSP 554 Big Data Technologies - Fraud Detection Project

**Role:** Quality Assurance & Documentation Lead  
**Timeline:** November 18 - December 10, 2025  
**Primary Focus:** Data profiling, testing framework, adversarial defenses, technical report, presentation

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

1. **Data Profiling & Quality Assurance**

   - Comprehensive data profiling with statistical summaries
   - Data quality validation and integrity checks
   - Missing values detection and handling
   - Correlation analysis and distribution visualizations

2. **Testing Framework Development**

   - Unit tests for all pipeline components
   - Integration tests for end-to-end pipeline
   - Test data generation strategies
   - Mock S3 and EMR for local testing

3. **Adversarial Defense Implementation**

   - Feature clamping utilities
   - Ensemble voting mechanisms
   - Input validation and sanitization
   - Adversarial example detection

4. **Documentation & Reporting**

   - Technical report compilation
   - Architecture diagrams
   - Presentation materials
   - Code documentation and runbooks

5. **Operations & Security**
   - Operational runbooks
   - Troubleshooting guides
   - Security audit and IAM policies
   - CI/CD pipeline setup (optional)

### Success Metrics

- ✅ Complete data profiling report generated
- ✅ Test coverage > 80%
- ✅ All tests pass before merge
- ✅ Technical report complete with all sections
- ✅ Presentation slides finalized
- ✅ Security audit completed

### Integration with Team

- **Receives from Rohit:** Pipeline components for testing, model artifacts
- **Receives from Ansh:** Evaluation metrics, model performance data
- **Provides to Rohit:** Data profiling reports, quality validation
- **Provides to Ansh:** Test fixtures, quality metrics
- **Coordinates:** Documentation, final report compilation

---

## Phase-by-Phase Execution Plan

### Phase 1: Data Profiling & Quality Checks (Nov 18-22)

**Goal:** Complete data profiling and establish quality baseline

#### Day 1 (Nov 18): Setup & Basic Profiling

**Tasks:**

1. Set up Git branch
2. Install profiling dependencies
3. Implement basic profiling functions

#### Day 2-3 (Nov 19-20): Comprehensive Profiling

**Tasks:**

1. Statistical summaries
2. Missing values detection
3. Correlation analysis
4. Distribution visualizations

#### Day 4-5 (Nov 21-22): Quality Validation & Reporting

**Tasks:**

1. Quality checks implementation
2. HTML report generation
3. Data integrity verification

**Deliverable:** Complete profiling report and quality validation

---

### Phase 2: Testing Framework (Nov 23-27)

**Goal:** Develop comprehensive testing framework

#### Day 1-2 (Nov 23-24): Unit Tests

**Tasks:**

1. Set up pytest with Spark
2. Unit tests for pipeline components
3. Mock S3 and EMR utilities

#### Day 3-4 (Nov 25-26): Integration Tests

**Tasks:**

1. End-to-end pipeline tests
2. Test data generation
3. Synthetic fraud examples

#### Day 5 (Nov 27): CI/CD Setup (Optional)

**Tasks:**

1. GitHub Actions configuration
2. Automated test runs
3. Test coverage reporting

**Deliverable:** Complete testing framework with >80% coverage

---

### Phase 3: Adversarial Defenses (Nov 28-Dec 3)

**Goal:** Implement and test defense mechanisms

#### Day 1-2 (Nov 28-29): Defense Implementation

**Tasks:**

1. Feature clamping utilities
2. Ensemble voting mechanism
3. Input validation functions

#### Day 3-4 (Nov 30-Dec 1): Defense Testing

**Tasks:**

1. Robustness testing
2. Defense evaluation
3. Performance impact analysis

#### Day 5-6 (Dec 2-3): Documentation

**Tasks:**

1. Defense documentation
2. Test suite for adversarial scenarios
3. Integration with evaluation framework

**Deliverable:** Complete defense implementation with tests

---

### Phase 4: Streaming QA & Operations (Dec 4-6)

**Goal:** Quality assurance for streaming and operational documentation

#### Day 1-2 (Dec 4-5): Streaming QA

**Tasks:**

1. Checkpoint validation
2. Fault injection testing
3. Performance monitoring

#### Day 3 (Dec 6): Operations Documentation

**Tasks:**

1. Runbooks creation
2. Troubleshooting guides
3. Security audit

**Deliverable:** Operational documentation complete

---

### Phase 5: Report & Presentation (Dec 7-10)

**Goal:** Finalize technical report and presentation

#### Day 1-2 (Dec 7-8): Report Compilation

**Tasks:**

1. Literature review consolidation
2. Methodology section
3. Results aggregation
4. References management

#### Day 3 (Dec 9): Presentation Preparation

**Tasks:**

1. Slide deck creation
2. Speaker notes
3. Demo scenario scripting

#### Day 4 (Dec 10): Final Polish

**Tasks:**

1. Report review and editing
2. Presentation rehearsal
3. Final repository cleanup

**Deliverable:** Complete technical report and presentation

---

## File Structure

```
src/hussain/
├── __init__.py
├── profiling/
│   ├── __init__.py
│   ├── data_profiler.py              # Main profiling implementation
│   ├── quality_checks.py              # Quality validation
│   └── report_generator.py            # HTML report generation
├── testing/
│   ├── __init__.py
│   ├── test_pipelines.py              # Pipeline tests
│   ├── test_adversarial.py            # Adversarial tests
│   ├── conftest.py                    # Pytest configuration
│   └── test_data_generator.py         # Test data generation
├── defenses/
│   ├── __init__.py
│   ├── feature_clamping.py            # Feature clamping defense
│   ├── ensemble_voting.py            # Ensemble defense
│   └── input_validation.py            # Input validation
├── documentation/
│   ├── architecture_diagrams.py       # Diagrams as code
│   ├── report_sections/
│   │   ├── introduction.md
│   │   ├── methodology.md
│   │   └── results.md
│   └── presentation_outline.md
└── operations/
    ├── runbook.md                     # Operational runbook
    ├── troubleshooting.md             # Troubleshooting guide
    └── security_audit.py              # Security audit script
```

---

## Setup & Configuration

### Dependencies Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install core dependencies
pip install pyspark==3.5.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Install profiling tools
pip install ydata-profiling==4.6.1
pip install pandas-profiling==3.6.6  # Alternative

# Install testing frameworks
pip install pytest==7.4.0
pip install pytest-spark==0.6.0
pip install pytest-cov==4.1.0
pip install moto==4.2.0  # For mocking AWS services

# Install documentation tools
pip install diagrams==0.23.4  # Diagrams as code
pip install sphinx==7.1.2  # Documentation generation

# Install security tools
pip install bandit==1.7.5  # Security linter
pip install safety==2.3.5  # Dependency vulnerability scanner

# Save to requirements.txt
pip freeze > requirements.txt
```

### Pytest Configuration

**File:** `tests/conftest.py`

```python
"""
Pytest configuration for Spark testing.
"""
import pytest
from pyspark.sql import SparkSession
import os

@pytest.fixture(scope="session")
def spark():
    """Create Spark session for testing."""
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-spark") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    yield spark

    spark.stop()

@pytest.fixture
def sample_fraud_data(spark):
    """Create sample fraud detection data for testing."""
    from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

    schema = StructType([
        StructField("Time", DoubleType(), True),
        StructField("V1", DoubleType(), True),
        StructField("V2", DoubleType(), True),
        StructField("Amount", DoubleType(), True),
        StructField("Class", IntegerType(), True)
    ])

    data = [
        (0.0, 1.0, 2.0, 100.0, 0),
        (1.0, 1.5, 2.5, 200.0, 0),
        (2.0, -1.0, -2.0, 5000.0, 1),  # Fraud case
        (3.0, 0.5, 1.0, 50.0, 0)
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def mock_s3(monkeypatch):
    """Mock S3 for local testing."""
    import boto3
    from moto import mock_s3

    with mock_s3():
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-bucket')
        yield s3_client
```

### GitHub Actions CI/CD

**File:** `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, feature/*]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

      - name: Security scan
        run: |
          pip install bandit safety
          bandit -r src/
          safety check
```

---

## Code Implementations

### Phase 1: Data Profiling

**File:** `src/hussain/profiling/data_profiler.py`

```python
"""
Comprehensive data profiling for credit card fraud detection dataset.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from src.common.config import config
from src.common.schema_validator import get_credit_card_schema

class CreditCardDataProfiler:
    """Data profiler for credit card fraud detection."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def profile_dataset(self, df: DataFrame) -> Dict:
        """
        Generate comprehensive data profile.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with profiling results
        """
        profile = {
            "basic_statistics": self._basic_statistics(df),
            "missing_values": self._missing_values(df),
            "fraud_statistics": self._fraud_statistics(df),
            "correlation_analysis": self._correlation_analysis(df),
            "distribution_statistics": self._distribution_statistics(df),
            "data_quality": self._data_quality_checks(df)
        }

        return profile

    def _basic_statistics(self, df: DataFrame) -> Dict:
        """Calculate basic statistics."""
        total_count = df.count()
        num_features = len(df.columns)

        stats = {
            "total_records": total_count,
            "num_features": num_features,
            "feature_names": df.columns
        }

        return stats

    def _missing_values(self, df: DataFrame) -> Dict:
        """Detect missing values."""
        missing_counts = {}
        total_count = df.count()

        for col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            missing_counts[col] = {
                "count": null_count,
                "percentage": (null_count / total_count) * 100 if total_count > 0 else 0
            }

        return missing_counts

    def _fraud_statistics(self, df: DataFrame) -> Dict:
        """Calculate fraud-related statistics."""
        total_count = df.count()
        fraud_count = df.filter(F.col("Class") == 1).count()
        normal_count = df.filter(F.col("Class") == 0).count()

        fraud_rate = (fraud_count / total_count) * 100 if total_count > 0 else 0

        # Amount statistics by class
        fraud_amount_stats = df.filter(F.col("Class") == 1).agg(
            F.avg("Amount").alias("avg_amount"),
            F.stddev("Amount").alias("std_amount"),
            F.min("Amount").alias("min_amount"),
            F.max("Amount").alias("max_amount")
        ).collect()[0]

        normal_amount_stats = df.filter(F.col("Class") == 0).agg(
            F.avg("Amount").alias("avg_amount"),
            F.stddev("Amount").alias("std_amount"),
            F.min("Amount").alias("min_amount"),
            F.max("Amount").alias("max_amount")
        ).collect()[0]

        stats = {
            "total_transactions": total_count,
            "fraud_transactions": fraud_count,
            "normal_transactions": normal_count,
            "fraud_rate_percentage": fraud_rate,
            "fraud_amount_stats": {
                "avg": float(fraud_amount_stats["avg_amount"]),
                "std": float(fraud_amount_stats["std_amount"]),
                "min": float(fraud_amount_stats["min_amount"]),
                "max": float(fraud_amount_stats["max_amount"])
            },
            "normal_amount_stats": {
                "avg": float(normal_amount_stats["avg_amount"]),
                "std": float(normal_amount_stats["std_amount"]),
                "min": float(normal_amount_stats["min_amount"]),
                "max": float(normal_amount_stats["max_amount"])
            }
        }

        return stats

    def _correlation_analysis(self, df: DataFrame) -> Dict:
        """Perform correlation analysis."""
        # Convert to Pandas for correlation (for smaller datasets)
        # For larger datasets, use Spark's correlation function

        # Select numeric columns
        numeric_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

        # Calculate correlation with Class
        correlations = {}
        for col in numeric_cols:
            try:
                corr = df.stat.corr(col, "Class")
                correlations[col] = float(corr) if corr is not None else 0.0
            except:
                correlations[col] = 0.0

        # Sort by absolute correlation
        sorted_correlations = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "class_correlations": sorted_correlations,
            "top_correlated_features": list(sorted_correlations.keys())[:10]
        }

    def _distribution_statistics(self, df: DataFrame) -> Dict:
        """Calculate distribution statistics."""
        numeric_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

        distribution_stats = {}
        for col in numeric_cols[:5]:  # Sample first 5 for performance
            stats = df.select(
                F.mean(col).alias("mean"),
                F.stddev(col).alias("std"),
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.percentile_approx(col, 0.25).alias("q1"),
                F.percentile_approx(col, 0.5).alias("median"),
                F.percentile_approx(col, 0.75).alias("q3")
            ).collect()[0]

            distribution_stats[col] = {
                "mean": float(stats["mean"]) if stats["mean"] else 0.0,
                "std": float(stats["std"]) if stats["std"] else 0.0,
                "min": float(stats["min"]) if stats["min"] else 0.0,
                "max": float(stats["max"]) if stats["max"] else 0.0,
                "q1": float(stats["q1"]) if stats["q1"] else 0.0,
                "median": float(stats["median"]) if stats["median"] else 0.0,
                "q3": float(stats["q3"]) if stats["q3"] else 0.0
            }

        return distribution_stats

    def _data_quality_checks(self, df: DataFrame) -> Dict:
        """Perform data quality checks."""
        checks = {
            "has_missing_values": False,
            "has_duplicates": False,
            "has_negative_amounts": False,
            "has_invalid_class_labels": False,
            "data_quality_passed": True
        }

        # Check missing values
        for col in df.columns:
            if df.filter(F.col(col).isNull()).count() > 0:
                checks["has_missing_values"] = True
                checks["data_quality_passed"] = False
                break

        # Check duplicates (if Time column exists)
        if "Time" in df.columns:
            duplicate_count = df.groupBy(df.columns).count().filter(F.col("count") > 1).count()
            if duplicate_count > 0:
                checks["has_duplicates"] = True

        # Check negative amounts
        if "Amount" in df.columns:
            negative_amounts = df.filter(F.col("Amount") < 0).count()
            if negative_amounts > 0:
                checks["has_negative_amounts"] = True

        # Check class labels (should be 0 or 1)
        if "Class" in df.columns:
            invalid_labels = df.filter(~F.col("Class").isin([0, 1])).count()
            if invalid_labels > 0:
                checks["has_invalid_class_labels"] = True
                checks["data_quality_passed"] = False

        return checks

    def save_profile_to_s3(self, profile: Dict, output_path: str = None):
        """Save profiling results to S3."""
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = f"{config.S3_BUCKET}/profiling/metadata_{timestamp}.json"

        import boto3
        import json

        s3_client = boto3.client('s3')
        bucket, key = output_path.replace("s3://", "").split("/", 1)

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(profile, indent=2),
            ContentType="application/json"
        )

        print(f"Profile saved to {output_path}")

# Usage
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session

    spark = create_spark_session("DataProfiling")

    # Load data
    df = spark.read.csv(config.S3_RAW_DATA_PATH, header=True, schema=get_credit_card_schema())

    # Profile data
    profiler = CreditCardDataProfiler(spark)
    profile = profiler.profile_dataset(df)

    # Print summary
    print("\n=== Data Profile Summary ===")
    print(f"Total Records: {profile['basic_statistics']['total_records']:,}")
    print(f"Fraud Rate: {profile['fraud_statistics']['fraud_rate_percentage']:.2f}%")
    print(f"Data Quality Passed: {profile['data_quality']['data_quality_passed']}")

    # Save to S3
    profiler.save_profile_to_s3(profile)

    spark.stop()
```

**File:** `src/hussain/profiling/quality_checks.py`

```python
"""
Data quality validation checks.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Dict, List

class DataQualityChecker:
    """Data quality validation framework."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def run_all_checks(self, df: DataFrame) -> Dict:
        """Run all quality checks."""
        checks = {
            "schema_validation": self.check_schema(df),
            "missing_values": self.check_missing_values(df),
            "data_types": self.check_data_types(df),
            "value_ranges": self.check_value_ranges(df),
            "class_distribution": self.check_class_distribution(df),
            "overall_status": "PASSED"
        }

        # Determine overall status
        for check_name, check_result in checks.items():
            if check_name != "overall_status" and isinstance(check_result, dict):
                if not check_result.get("passed", True):
                    checks["overall_status"] = "FAILED"
                    break

        return checks

    def check_schema(self, df: DataFrame) -> Dict:
        """Validate schema matches expected structure."""
        expected_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
        actual_cols = df.columns

        missing_cols = set(expected_cols) - set(actual_cols)
        extra_cols = set(actual_cols) - set(expected_cols)

        return {
            "passed": len(missing_cols) == 0 and len(extra_cols) == 0,
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols)
        }

    def check_missing_values(self, df: DataFrame) -> Dict:
        """Check for missing values."""
        missing_counts = {}
        total_count = df.count()

        for col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            if null_count > 0:
                missing_counts[col] = {
                    "count": null_count,
                    "percentage": (null_count / total_count) * 100
                }

        return {
            "passed": len(missing_counts) == 0,
            "missing_values": missing_counts
        }

    def check_data_types(self, df: DataFrame) -> Dict:
        """Validate data types."""
        issues = []

        # Check Class is integer
        if "Class" in df.columns:
            if str(df.schema["Class"].dataType) != "IntegerType":
                issues.append("Class column should be IntegerType")

        # Check Amount is numeric
        if "Amount" in df.columns:
            if "DoubleType" not in str(df.schema["Amount"].dataType):
                issues.append("Amount column should be DoubleType")

        return {
            "passed": len(issues) == 0,
            "issues": issues
        }

    def check_value_ranges(self, df: DataFrame) -> Dict:
        """Check value ranges for key columns."""
        issues = []

        # Check Amount >= 0
        if "Amount" in df.columns:
            negative_amounts = df.filter(F.col("Amount") < 0).count()
            if negative_amounts > 0:
                issues.append(f"Found {negative_amounts} negative amounts")

        # Check Class in [0, 1]
        if "Class" in df.columns:
            invalid_classes = df.filter(~F.col("Class").isin([0, 1])).count()
            if invalid_classes > 0:
                issues.append(f"Found {invalid_classes} invalid class labels")

        return {
            "passed": len(issues) == 0,
            "issues": issues
        }

    def check_class_distribution(self, df: DataFrame) -> Dict:
        """Check class distribution for imbalance."""
        total_count = df.count()
        fraud_count = df.filter(F.col("Class") == 1).count()
        fraud_rate = (fraud_count / total_count) * 100

        # Expected fraud rate: ~0.17%
        expected_rate = 0.17
        tolerance = 0.05  # 5% tolerance

        passed = abs(fraud_rate - expected_rate) < tolerance

        return {
            "passed": passed,
            "fraud_rate": fraud_rate,
            "expected_rate": expected_rate,
            "tolerance": tolerance
        }

# Usage
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session
    from src.common.config import config

    spark = create_spark_session("QualityChecks")

    df = spark.read.csv(config.S3_RAW_DATA_PATH, header=True)

    checker = DataQualityChecker(spark)
    results = checker.run_all_checks(df)

    print("\n=== Quality Check Results ===")
    print(f"Overall Status: {results['overall_status']}")
    for check_name, check_result in results.items():
        if check_name != "overall_status":
            print(f"{check_name}: {'PASSED' if check_result.get('passed', False) else 'FAILED'}")

    spark.stop()
```

**File:** `src/hussain/profiling/report_generator.py`

```python
"""
Generate HTML profiling report.
"""
from pyspark.sql import SparkSession
import pandas as pd
from ydata_profiling import ProfileReport
from src.hussain.profiling.data_profiler import CreditCardDataProfiler

def generate_html_report(spark: SparkSession, df, output_path: str = "profiling_report.html"):
    """
    Generate HTML profiling report using ydata-profiling.

    Args:
        spark: Spark session
        df: DataFrame to profile
        output_path: Path to save HTML report
    """
    # Convert sample to Pandas (for ydata-profiling)
    # For large datasets, sample first
    sample_size = min(100000, df.count())
    sample_df = df.sample(False, sample_size / df.count(), seed=42).limit(sample_size)

    pandas_df = sample_df.toPandas()

    # Generate profile
    profile = ProfileReport(
        pandas_df,
        title="Credit Card Fraud Detection - Data Profile",
        explorative=True,
        minimal=False
    )

    # Save report
    profile.to_file(output_path)
    print(f"HTML report saved to {output_path}")

    # Also upload to S3
    import boto3
    from src.common.config import config
    from datetime import datetime

    s3_client = boto3.client('s3')
    timestamp = datetime.now().strftime("%Y%m%d")
    s3_key = f"profiling/report_{timestamp}.html"

    s3_client.upload_file(output_path, config.S3_BUCKET, s3_key)
    print(f"Report uploaded to s3://{config.S3_BUCKET}/{s3_key}")

# Usage
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session
    from src.common.config import config

    spark = create_spark_session("ReportGeneration")

    df = spark.read.csv(config.S3_RAW_DATA_PATH, header=True)

    generate_html_report(spark, df, "profiling_report.html")

    spark.stop()
```

### Phase 2: Testing Framework

**File:** `tests/test_pipelines.py`

```python
"""
Unit tests for pipeline components.
"""
import pytest
from pyspark.sql import SparkSession
from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline
from src.rohit.pipelines.logistic_regression import LogisticRegressionPipeline

def test_base_pipeline_load_data(spark, sample_fraud_data):
    """Test data loading in base pipeline."""
    pipeline = BaseFraudDetectionPipeline(spark)

    # Test with sample data
    df = sample_fraud_data
    assert df.count() > 0
    assert "Class" in df.columns

def test_base_pipeline_train_test_split(spark, sample_fraud_data):
    """Test train-test split."""
    pipeline = BaseFraudDetectionPipeline(spark)
    df = sample_fraud_data.withColumnRenamed("Class", "label")

    train_df, test_df = pipeline.train_test_split(df, train_ratio=0.8)

    assert train_df.count() > 0
    assert test_df.count() > 0
    assert train_df.count() + test_df.count() == df.count()

def test_logistic_regression_pipeline(spark, sample_fraud_data):
    """Test Logistic Regression pipeline."""
    pipeline = LogisticRegressionPipeline(spark)
    df = sample_fraud_data.withColumnRenamed("Class", "label")

    # Add missing V columns for full test
    for i in range(3, 29):
        df = df.withColumn(f"V{i}", df.V1)  # Simple fill for testing

    train_df, test_df = pipeline.train_test_split(df, train_ratio=0.8)

    # Train model
    model = pipeline.train(train_df, save_model=False)

    # Evaluate
    metrics = pipeline.evaluate(model, test_df)

    assert "auroc" in metrics
    assert 0 <= metrics["auroc"] <= 1

# Run tests: pytest tests/test_pipelines.py -v
```

**File:** `tests/test_adversarial.py`

```python
"""
Tests for adversarial robustness.
"""
import pytest
from pyspark.sql import SparkSession
from src.ansh.adversarial.fgsm_attack import FGSMAttack
from pyspark.ml import PipelineModel

def test_fgsm_attack_generation(spark, sample_fraud_data):
    """Test FGSM attack generation."""
    # Load or create a simple model for testing
    # (In practice, load from S3)

    # Mock model for testing
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import LogisticRegression

    df = sample_fraud_data.withColumnRenamed("Class", "label")
    for i in range(3, 29):
        df = df.withColumn(f"V{i}", df.V1)

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    pipeline = Pipeline(stages=[assembler, lr])
    model = pipeline.fit(df)

    # Test FGSM attack
    attack = FGSMAttack(spark, model, epsilon=0.1)
    adversarial_df = attack.generate_adversarial_examples(df, feature_cols)

    assert adversarial_df.count() == df.count()
    assert len(adversarial_df.columns) > 0

# Run tests: pytest tests/test_adversarial.py -v
```

### Phase 3: Adversarial Defenses

**File:** `src/hussain/defenses/feature_clamping.py`

```python
"""
Feature clamping defense against adversarial attacks.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Dict

class FeatureClamping:
    """Clamp features to valid ranges to defend against adversarial attacks."""

    def __init__(self, feature_ranges: Dict[str, tuple]):
        """
        Initialize feature clamping.

        Args:
            feature_ranges: Dictionary of {feature_name: (min, max)}
        """
        self.feature_ranges = feature_ranges

    def clamp_features(self, df: DataFrame) -> DataFrame:
        """
        Clamp features to valid ranges.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with clamped features
        """
        clamped_df = df

        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                clamped_df = clamped_df.withColumn(
                    feature,
                    F.when(F.col(feature) < min_val, min_val)
                     .when(F.col(feature) > max_val, max_val)
                     .otherwise(F.col(feature))
                )

        return clamped_df

    @staticmethod
    def get_default_ranges() -> Dict[str, tuple]:
        """Get default feature ranges based on dataset statistics."""
        # These would be calculated from training data
        ranges = {}

        # V1-V28: Typically in range [-10, 10] after PCA
        for i in range(1, 29):
            ranges[f"V{i}"] = (-10.0, 10.0)

        # Amount: Based on dataset, typically [0, 50000]
        ranges["Amount"] = (0.0, 50000.0)

        return ranges

# Usage
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session

    spark = create_spark_session("FeatureClamping")

    # Load data
    df = spark.read.csv("s3://bucket/processed/test/", header=True)

    # Create clamps
    clamps = FeatureClamping(FeatureClamping.get_default_ranges())
    clamped_df = clamps.clamp_features(df)

    print("Features clamped successfully")

    spark.stop()
```

**File:** `src/hussain/defenses/ensemble_voting.py`

```python
"""
Ensemble voting defense mechanism.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from typing import List

class EnsembleVoting:
    """Ensemble voting for robust predictions."""

    def __init__(self, models: List[PipelineModel], voting_strategy: str = "majority"):
        """
        Initialize ensemble voting.

        Args:
            models: List of trained models
            voting_strategy: "majority" or "weighted"
        """
        self.models = models
        self.voting_strategy = voting_strategy

    def predict(self, df: DataFrame) -> DataFrame:
        """
        Make ensemble predictions.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with ensemble predictions
        """
        predictions_list = []

        # Get predictions from each model
        for i, model in enumerate(self.models):
            pred_df = model.transform(df)
            pred_df = pred_df.withColumnRenamed("prediction", f"pred_{i}")
            pred_df = pred_df.withColumnRenamed("probability", f"prob_{i}")
            predictions_list.append(pred_df.select("pred_{}".format(i), "prob_{}".format(i)))

        # Combine predictions
        # (Simplified - in practice, need to join properly)
        ensemble_df = df

        if self.voting_strategy == "majority":
            # Majority voting
            for i in range(len(self.models)):
                ensemble_df = ensemble_df.join(
                    predictions_list[i],
                    how="inner"
                )

            # Calculate majority prediction
            pred_cols = [f"pred_{i}" for i in range(len(self.models))]
            ensemble_df = ensemble_df.withColumn(
                "ensemble_prediction",
                F.when(sum(F.col(f"pred_{i}") for i in range(len(self.models))) > len(self.models) / 2, 1)
                .otherwise(0)
            )

        return ensemble_df

# Usage
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session
    from src.common.config import config
    from pyspark.ml import PipelineModel

    spark = create_spark_session("EnsembleVoting")

    # Load multiple models
    models = [
        PipelineModel.load(f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/"),
        PipelineModel.load(f"{config.S3_MODELS_PATH}/random_forest/v1/model/"),
        PipelineModel.load(f"{config.S3_MODELS_PATH}/gbt_classifier/v1/model/")
    ]

    # Create ensemble
    ensemble = EnsembleVoting(models, voting_strategy="majority")

    # Load test data
    test_df = spark.read.parquet(f"{config.S3_PROCESSED_PATH}/test/")

    # Make ensemble predictions
    ensemble_predictions = ensemble.predict(test_df)

    print("Ensemble predictions generated")

    spark.stop()
```

### Phase 4: Operations Documentation

**File:** `src/hussain/operations/runbook.md`

````markdown
# Operational Runbook - Fraud Detection Pipeline

## Cluster Management

### Starting EMR Cluster

```bash
./src/rohit/infrastructure/emr_setup.sh fraud-detection-cluster $BUCKET_NAME
```
````

### Checking Cluster Status

```bash
aws emr describe-cluster --cluster-id <cluster-id>
```

### Terminating Cluster

```bash
aws emr terminate-clusters --cluster-ids <cluster-id>
```

## Data Pipeline

### Running Data Profiling

```bash
spark-submit src/hussain/profiling/data_profiler.py
```

### Training Models

```bash
spark-submit src/rohit/pipelines/logistic_regression.py
```

### Running Evaluation

```bash
spark-submit src/ansh/evaluation/metrics.py
```

## Troubleshooting

### Cluster Not Starting

- Check IAM roles and permissions
- Verify security groups
- Check CloudWatch logs

### Out of Memory Errors

- Increase executor memory in spark_config.py
- Reduce data size for testing
- Use data sampling

### S3 Access Denied

- Verify IAM policies
- Check bucket permissions
- Verify credentials

````

### Phase 5: Technical Report Template

**File:** `src/hussain/documentation/report_sections/introduction.md`

```markdown
# Introduction

## Problem Statement

Credit card fraud is a significant challenge in the financial industry, with billions of dollars lost annually to fraudulent transactions. Traditional rule-based fraud detection systems are limited in their ability to adapt to evolving fraud patterns. Machine learning approaches, particularly those leveraging big data technologies, offer promising solutions for real-time fraud detection at scale.

## Dataset Description

The Credit Card Fraud Detection dataset from Kaggle contains 284,807 transactions, of which 492 (0.17%) are fraudulent. The dataset includes 30 features: 28 principal components (V1-V28) derived from PCA transformation, transaction amount, and time. The class imbalance presents a significant challenge for model training and evaluation.

## Objectives

1. Develop scalable fraud detection models using Apache Spark MLlib on AWS EMR
2. Evaluate model performance using comprehensive metrics (AUROC, AUPRC, F-score)
3. Assess adversarial robustness of models against FGSM attacks
4. Implement defense mechanisms (ensemble voting, feature clamping)
5. Integrate models into streaming pipeline for real-time detection
6. Compare batch and streaming performance

## Contributions

This project demonstrates:
- End-to-end ML pipeline on AWS EMR with Spark MLlib
- Handling of imbalanced datasets in distributed computing
- Adversarial robustness evaluation for tabular data
- Streaming ML integration for fraud detection
````

---

## Integration Points

### Providing Profiling Data to Rohit

```python
# Save profiling metadata
profiler = CreditCardDataProfiler(spark)
profile = profiler.profile_dataset(df)
profiler.save_profile_to_s3(profile)

# Rohit can load it:
import json
import boto3

s3_client = boto3.client('s3')
response = s3_client.get_object(Bucket=config.S3_BUCKET, Key="profiling/metadata_latest.json")
profiling_metadata = json.loads(response['Body'].read().decode('utf-8'))
```

### Providing Test Fixtures to Ansh

```python
# Export test data generator
from src.hussain.testing.test_data_generator import generate_synthetic_fraud_data

# Ansh can use for evaluation testing
test_df = generate_synthetic_fraud_data(spark, n_samples=1000, fraud_rate=0.17)
```

### Receiving Models from Rohit

```python
# Test Rohit's models
from pyspark.ml import PipelineModel
from src.common.config import config

model = PipelineModel.load(f"{config.S3_MODELS_PATH}/logistic_regression/v1/model/")
# Run tests...
```

---

## Quality Gates

### Test Coverage

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Check coverage
# Target: >80% coverage
```

### Code Quality

```bash
# Run linting
flake8 src/
black --check src/

# Security scan
bandit -r src/
safety check
```

### Documentation Completeness

- [ ] All functions have docstrings
- [ ] README.md is complete
- [ ] Architecture diagrams included
- [ ] Runbook is comprehensive

---

## Daily Task Breakdown

### Phase 1 (Nov 18-22)

| Day    | Task                   | Hours | Deliverable        |
| ------ | ---------------------- | ----- | ------------------ |
| Nov 18 | Setup, basic profiling | 4     | Profiling skeleton |
| Nov 19 | Statistical summaries  | 6     | Complete profiling |
| Nov 20 | Quality checks         | 4     | Quality validation |
| Nov 21 | Report generation      | 4     | HTML report        |
| Nov 22 | Integration            | 4     | Profiling complete |

### Phase 2 (Nov 23-27)

| Day    | Task                 | Hours | Deliverable    |
| ------ | -------------------- | ----- | -------------- |
| Nov 23 | Pytest setup         | 4     | Test framework |
| Nov 24 | Unit tests           | 6     | Pipeline tests |
| Nov 25 | Integration tests    | 4     | E2E tests      |
| Nov 26 | Test data generation | 4     | Synthetic data |
| Nov 27 | CI/CD setup          | 4     | GitHub Actions |

### Phase 3 (Nov 28-Dec 3)

| Day     | Task             | Hours | Deliverable          |
| ------- | ---------------- | ----- | -------------------- |
| Nov 28  | Feature clamping | 4     | Clamping defense     |
| Nov 29  | Ensemble voting  | 4     | Ensemble defense     |
| Nov 30  | Defense testing  | 6     | Defense evaluation   |
| Dec 1   | Input validation | 4     | Validation functions |
| Dec 2-3 | Documentation    | 4     | Defense docs         |

### Phase 4 (Dec 4-6)

| Day   | Task            | Hours | Deliverable     |
| ----- | --------------- | ----- | --------------- |
| Dec 4 | Streaming QA    | 6     | Streaming tests |
| Dec 5 | Operations docs | 4     | Runbooks        |
| Dec 6 | Security audit  | 4     | Security report |

### Phase 5 (Dec 7-10)

| Day    | Task              | Hours | Deliverable      |
| ------ | ----------------- | ----- | ---------------- |
| Dec 7  | Report writing    | 8     | Report draft     |
| Dec 8  | Report completion | 6     | Final report     |
| Dec 9  | Presentation      | 6     | Slides ready     |
| Dec 10 | Final polish      | 4     | Submission ready |

---

## Checkpoint Questions

### After Phase 1

- [ ] Can you generate profiling report?
- [ ] Are quality checks passing?
- [ ] Is data ready for modeling?

### After Phase 2

- [ ] Is test coverage >80%?
- [ ] Do all tests pass?
- [ ] Is CI/CD working?

### After Phase 3

- [ ] Are defenses implemented?
- [ ] Do defenses improve robustness?
- [ ] Are defense tests passing?

### After Phase 4

- [ ] Is streaming QA complete?
- [ ] Are runbooks comprehensive?
- [ ] Is security audit done?

### After Phase 5

- [ ] Is technical report complete?
- [ ] Are presentation slides ready?
- [ ] Is documentation polished?

---

## Pro Tips

1. **Profiling Early:** Profile data before modeling to catch issues
2. **Test Coverage:** Aim for >80% but focus on critical paths
3. **Documentation:** Write docs as you code, not after
4. **Security:** Run security scans regularly
5. **Reproducibility:** Use fixed seeds everywhere
6. **Version Control:** Commit frequently with clear messages
7. **Collaboration:** Review team members' code regularly
8. **Performance:** Profile code to find bottlenecks
9. **Error Handling:** Test error scenarios, not just happy paths
10. **Communication:** Update team on blockers immediately

---

**Document Version:** 1.0  
**Last Updated:** November 18, 2025  
**Owner:** Hussain Bin Yousuf
