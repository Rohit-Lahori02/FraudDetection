"""
Tests for pipeline functionality.
"""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml import PipelineModel
from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline
from src.rohit.pipelines.logistic_regression import LogisticRegressionPipeline
from src.rohit.pipelines.random_forest import RandomForestPipeline
from src.rohit.pipelines.gbt_classifier import GBTPipeline
from src.common.spark_session import create_spark_session
from src.common.schema_validator import get_credit_card_schema


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for testing."""
    spark_session = create_spark_session("TestSession", master="local[2]")
    yield spark_session
    spark_session.stop()


@pytest.fixture
def sample_data(spark):
    """Create sample DataFrame for testing."""
    schema = get_credit_card_schema()
    
    # Create sample data
    data = []
    for i in range(100):
        row = [i]  # id
        # Add V1-V28 features
        for j in range(1, 29):
            row.append(float(i * 0.1 + j * 0.01))
        # Add Amount
        row.append(float(100.0 + i))
        # Add Class (alternating 0 and 1)
        row.append(i % 2)
        data.append(row)
    
    df = spark.createDataFrame(data, schema)
    # Rename Class to label
    df = df.withColumnRenamed("Class", "label")
    
    return df


def test_base_pipeline_init(spark):
    """Test BaseFraudDetectionPipeline initialization."""
    pipeline = BaseFraudDetectionPipeline(spark)
    assert pipeline is not None
    assert len(pipeline.feature_cols) == 29  # V1-V28 + Amount
    assert pipeline.label_col == "Class"


def test_base_pipeline_feature_pipeline(spark):
    """Test feature pipeline creation."""
    pipeline = BaseFraudDetectionPipeline(spark)
    feature_pipeline = pipeline.create_feature_pipeline()
    
    assert feature_pipeline is not None
    assert len(feature_pipeline.getStages()) == 2  # VectorAssembler + StandardScaler


def test_base_pipeline_train_test_split(spark, sample_data):
    """Test train/test split."""
    pipeline = BaseFraudDetectionPipeline(spark)
    train_df, test_df = pipeline.train_test_split(sample_data, train_ratio=0.8, seed=42)
    
    assert train_df is not None
    assert test_df is not None
    assert train_df.count() + test_df.count() == sample_data.count()


def test_logistic_regression_pipeline_init(spark):
    """Test LogisticRegressionPipeline initialization."""
    lr_pipeline = LogisticRegressionPipeline(spark)
    assert lr_pipeline is not None
    assert lr_pipeline.model_name == "logistic_regression"


def test_logistic_regression_create_pipeline(spark):
    """Test Logistic Regression pipeline creation."""
    lr_pipeline = LogisticRegressionPipeline(spark)
    pipeline = lr_pipeline.create_model_pipeline()
    
    assert pipeline is not None
    assert len(pipeline.getStages()) == 3  # VectorAssembler + StandardScaler + LR


def test_random_forest_pipeline_init(spark):
    """Test RandomForestPipeline initialization."""
    rf_pipeline = RandomForestPipeline(spark)
    assert rf_pipeline is not None
    assert rf_pipeline.model_name == "random_forest"


def test_random_forest_create_pipeline(spark):
    """Test Random Forest pipeline creation."""
    rf_pipeline = RandomForestPipeline(spark)
    pipeline = rf_pipeline.create_model_pipeline(numTrees=50, maxDepth=5)
    
    assert pipeline is not None
    assert len(pipeline.getStages()) == 3  # VectorAssembler + StandardScaler + RF


def test_gbt_pipeline_init(spark):
    """Test GBTPipeline initialization."""
    gbt_pipeline = GBTPipeline(spark)
    assert gbt_pipeline is not None
    assert gbt_pipeline.model_name == "gbt_classifier"


def test_gbt_create_pipeline(spark):
    """Test GBT pipeline creation."""
    gbt_pipeline = GBTPipeline(spark)
    pipeline = gbt_pipeline.create_model_pipeline(maxIter=50, maxDepth=3)
    
    assert pipeline is not None
    assert len(pipeline.getStages()) == 3  # VectorAssembler + StandardScaler + GBT


def test_logistic_regression_train(spark, sample_data):
    """Test Logistic Regression training."""
    lr_pipeline = LogisticRegressionPipeline(spark)
    train_df, test_df = lr_pipeline.train_test_split(sample_data, train_ratio=0.8, seed=42)
    
    # Train with small iterations for testing
    model = lr_pipeline.train(
        train_df,
        save_model=False,
        maxIter=10
    )
    
    assert model is not None
    assert isinstance(model, PipelineModel)


def test_random_forest_train(spark, sample_data):
    """Test Random Forest training."""
    rf_pipeline = RandomForestPipeline(spark)
    train_df, test_df = rf_pipeline.train_test_split(sample_data, train_ratio=0.8, seed=42)
    
    # Train with small parameters for testing
    model = rf_pipeline.train(
        train_df,
        save_model=False,
        numTrees=10,
        maxDepth=3
    )
    
    assert model is not None
    assert isinstance(model, PipelineModel)
    
    # Test feature importance
    importance = rf_pipeline.get_feature_importance(model)
    assert importance is not None
    assert len(importance) == 29  # Should have importance for all features


def test_gbt_train(spark, sample_data):
    """Test GBT training."""
    gbt_pipeline = GBTPipeline(spark)
    train_df, test_df = gbt_pipeline.train_test_split(sample_data, train_ratio=0.8, seed=42)
    
    # Train with small parameters for testing
    model = gbt_pipeline.train(
        train_df,
        save_model=False,
        maxIter=10,
        maxDepth=3
    )
    
    assert model is not None
    assert isinstance(model, PipelineModel)
    
    # Test feature importance
    importance = gbt_pipeline.get_feature_importance(model)
    assert importance is not None
    assert len(importance) == 29  # Should have importance for all features

