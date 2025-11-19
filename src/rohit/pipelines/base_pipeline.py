"""
Base pipeline class for fraud detection models.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import functions as F
from typing import List, Dict, Tuple
from src.common.config import config
from src.common.schema_validator import get_credit_card_schema


class BaseFraudDetectionPipeline:
    """Base class for fraud detection pipelines."""

    def __init__(self, spark: SparkSession):
        """
        Initialize base pipeline.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        self.feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        self.label_col = "Class"

    def load_data(self, s3_path: str = None) -> DataFrame:
        """
        Load credit card fraud dataset from S3 or local path.

        Args:
            s3_path: S3 path or local path to dataset (uses config if None)

        Returns:
            DataFrame with loaded data
        """
        if s3_path is None:
            s3_path = config.RAW_DATA_PATH

        schema = get_credit_card_schema()
        
        # Handle both local and S3 paths
        if s3_path.startswith("s3://") or s3_path.startswith("s3a://"):
            df = self.spark.read.schema(schema).csv(s3_path, header=True)
        else:
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
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            train_ratio: Proportion for training (default: 0.8)
            seed: Random seed (default: 42)

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
            version: Model version (default: "v1")
        """
        model_path = f"{config.MODELS_PATH}{model_name}/{version}/model/"

        # Save model
        model.write().overwrite().save(model_path)

        # Save metadata
        import json
        import boto3
        from datetime import datetime
        from botocore.exceptions import ClientError

        metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "feature_cols": self.feature_cols,
            "label_col": self.label_col
        }

        try:
            s3_client = boto3.client('s3')
            metadata_key = f"{config.MODELS_PATH}{model_name}/{version}/metadata.json"
            # Handle both s3://bucket/key and bucket/key formats
            if metadata_key.startswith("s3://"):
                bucket, key = metadata_key.replace("s3://", "").split("/", 1)
            else:
                bucket, key = metadata_key.split("/", 1)

            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(metadata, indent=2),
                ContentType="application/json"
            )

            print(f"Model saved to {model_path}")
            print(f"Metadata saved to {metadata_key}")
        except ClientError as e:
            print(f"Error saving metadata to S3: {e}")
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Note: S3 operation requires AWS credentials.")
            print(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_path: str) -> PipelineModel:
        """
        Load trained model from S3 or local path.

        Args:
            model_path: S3 path or local path to model

        Returns:
            Loaded PipelineModel
        """
        model = PipelineModel.load(model_path)
        print(f"Model loaded from {model_path}")
        return model


# Usage example
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session

    spark = create_spark_session("BasePipelineTest")

    # Create pipeline
    pipeline = BaseFraudDetectionPipeline(spark)

    # Load data
    df = pipeline.load_data()

    # Create feature pipeline
    feature_pipeline = pipeline.create_feature_pipeline()
    
    # Transform data
    processed_df = feature_pipeline.fit(df).transform(df)
    print(f"Processed {processed_df.count():,} records")

    # Split data
    train_df, test_df = pipeline.train_test_split(processed_df)

    print("\n=== Base Pipeline Test Complete ===")

    spark.stop()

