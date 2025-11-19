"""
Tests for evaluation metrics.
"""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row
from src.ansh.evaluation.metrics import FraudDetectionMetrics
from src.ansh.evaluation.cross_validation import FraudDetectionCrossValidator
from src.common.spark_session import create_spark_session


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for testing."""
    spark_session = create_spark_session("TestSession", master="local[2]")
    yield spark_session
    spark_session.stop()


@pytest.fixture
def sample_predictions(spark):
    """Create sample predictions DataFrame for testing."""
    # Create mock predictions with label, prediction, rawPrediction
    data = []
    for i in range(100):
        # Create rawPrediction vector (2 classes: [prob_class_0, prob_class_1])
        if i < 50:
            # First 50: class 0
            raw_pred = Vectors.dense([0.8, 0.2])
            label = 0
            prediction = 0
        else:
            # Last 50: class 1
            raw_pred = Vectors.dense([0.2, 0.8])
            label = 1
            prediction = 1
        
        data.append(Row(
            label=label,
            prediction=prediction,
            rawPrediction=raw_pred,
            probability=Vectors.dense([raw_pred[0], raw_pred[1]])
        ))
    
    schema = StructType([
        StructField("label", IntegerType(), True),
        StructField("prediction", IntegerType(), True),
        StructField("rawPrediction", VectorUDT(), True),
        StructField("probability", VectorUDT(), True)
    ])
    
    df = spark.createDataFrame(data, schema)
    return df


def test_metrics_calculator_init(spark):
    """Test FraudDetectionMetrics initialization."""
    metrics_calc = FraudDetectionMetrics(spark)
    assert metrics_calc is not None


def test_calculate_all_metrics(spark, sample_predictions):
    """Test calculation of all metrics."""
    metrics_calc = FraudDetectionMetrics(spark)
    metrics = metrics_calc.calculate_all_metrics(sample_predictions)
    
    assert metrics is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "specificity" in metrics
    assert "balanced_accuracy" in metrics
    assert "confusion_matrix" in metrics
    
    # Check that metrics are valid (between 0 and 1)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["auprc"] <= 1


def test_confusion_matrix(spark, sample_predictions):
    """Test confusion matrix calculation."""
    metrics_calc = FraudDetectionMetrics(spark)
    metrics = metrics_calc.calculate_all_metrics(sample_predictions)
    
    cm = metrics["confusion_matrix"]
    assert "tp" in cm
    assert "tn" in cm
    assert "fp" in cm
    assert "fn" in cm
    
    # For our sample data (perfect predictions), should have 50 TP and 50 TN
    assert cm["tp"] == 50
    assert cm["tn"] == 50
    assert cm["fp"] == 0
    assert cm["fn"] == 0


def test_cross_validator_init(spark):
    """Test FraudDetectionCrossValidator initialization."""
    cv_validator = FraudDetectionCrossValidator(spark, random_seed=42)
    assert cv_validator is not None


def test_stratified_split(spark):
    """Test stratified split functionality."""
    # Create sample data with class imbalance
    data = []
    for i in range(1000):
        label = 1 if i < 100 else 0  # 10% fraud rate
        data.append(Row(
            id=i,
            V1=float(i * 0.1),
            Amount=float(100.0 + i),
            label=label
        ))
    
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("V1", DoubleType(), True),
        StructField("Amount", DoubleType(), True),
        StructField("label", IntegerType(), True)
    ])
    
    df = spark.createDataFrame(data, schema)
    
    cv_validator = FraudDetectionCrossValidator(spark, random_seed=42)
    train_df, test_df = cv_validator.stratified_split(df, train_ratio=0.8)
    
    assert train_df is not None
    assert test_df is not None
    assert train_df.count() + test_df.count() == df.count()
    
    # Check that fraud rate is maintained
    train_fraud_rate = train_df.filter(train_df.label == 1).count() / train_df.count()
    test_fraud_rate = test_df.filter(test_df.label == 1).count() / test_df.count()
    
    # Should be approximately 10% in both splits
    assert 0.08 <= train_fraud_rate <= 0.12
    assert 0.08 <= test_fraud_rate <= 0.12

