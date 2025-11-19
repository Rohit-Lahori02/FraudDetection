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
        """
        Initialize Logistic Regression pipeline.

        Args:
            spark: SparkSession instance
        """
        super().__init__(spark)
        self.model_name = "logistic_regression"

    def create_model_pipeline(
        self,
        maxIter: int = 100,
        regParam: float = 0.01,
        elasticNetParam: float = 0.0
    ) -> Pipeline:
        """
        Create complete pipeline with Logistic Regression.

        Args:
            maxIter: Maximum number of iterations (default: 100)
            regParam: Regularization parameter (default: 0.01)
            elasticNetParam: Elastic net mixing parameter (default: 0.0 for L2)

        Returns:
            Pipeline with feature engineering and classifier
        """
        # Feature pipeline
        feature_pipeline = self.create_feature_pipeline()

        # Logistic Regression
        lr = LogisticRegression(
            featuresCol="scaled_features",
            labelCol="label",
            maxIter=maxIter,
            regParam=regParam,
            elasticNetParam=elasticNetParam
        )

        # Complete pipeline
        pipeline = Pipeline(stages=feature_pipeline.getStages() + [lr])

        return pipeline

    def train(
        self,
        train_df: DataFrame,
        save_model: bool = True,
        version: str = "v1",
        maxIter: int = 100,
        regParam: float = 0.01,
        elasticNetParam: float = 0.0
    ) -> PipelineModel:
        """
        Train Logistic Regression model.

        Args:
            train_df: Training DataFrame
            save_model: Whether to save model to S3
            version: Model version (default: "v1")
            maxIter: Maximum number of iterations
            regParam: Regularization parameter
            elasticNetParam: Elastic net mixing parameter

        Returns:
            Trained PipelineModel
        """
        print("Training Logistic Regression model...")

        # Create pipeline
        pipeline = self.create_model_pipeline(
            maxIter=maxIter,
            regParam=regParam,
            elasticNetParam=elasticNetParam
        )

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
    from src.common.spark_session import create_spark_session

    spark = create_spark_session("LogisticRegressionTraining")

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

