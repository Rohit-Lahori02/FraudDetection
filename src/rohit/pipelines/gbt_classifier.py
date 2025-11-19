"""
Gradient Boosted Trees pipeline for fraud detection.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline
from src.common.config import config


class GBTPipeline(BaseFraudDetectionPipeline):
    """Gradient Boosted Trees pipeline implementation."""

    def __init__(self, spark: SparkSession):
        """
        Initialize GBT pipeline.

        Args:
            spark: SparkSession instance
        """
        super().__init__(spark)
        self.model_name = "gbt_classifier"

    def create_model_pipeline(
        self,
        maxIter: int = 100,
        maxDepth: int = 5,
        stepSize: float = 0.1,
        seed: int = 42
    ) -> Pipeline:
        """
        Create complete pipeline with Gradient Boosted Trees.

        Args:
            maxIter: Maximum number of iterations (default: 100)
            maxDepth: Maximum depth of trees (default: 5)
            stepSize: Step size (learning rate) (default: 0.1)
            seed: Random seed (default: 42)

        Returns:
            Pipeline with feature engineering and classifier
        """
        feature_pipeline = self.create_feature_pipeline()

        gbt = GBTClassifier(
            featuresCol="scaled_features",
            labelCol="label",
            maxIter=maxIter,
            maxDepth=maxDepth,
            stepSize=stepSize,
            seed=seed
        )

        pipeline = Pipeline(stages=feature_pipeline.getStages() + [gbt])
        return pipeline

    def train(
        self,
        train_df: DataFrame,
        save_model: bool = True,
        version: str = "v1",
        maxIter: int = 100,
        maxDepth: int = 5,
        stepSize: float = 0.1,
        seed: int = 42
    ) -> PipelineModel:
        """
        Train Gradient Boosted Trees model.

        Args:
            train_df: Training DataFrame
            save_model: Whether to save model to S3
            version: Model version (default: "v1")
            maxIter: Maximum number of iterations
            maxDepth: Maximum depth of trees
            stepSize: Step size (learning rate)
            seed: Random seed

        Returns:
            Trained PipelineModel
        """
        print("Training Gradient Boosted Trees model...")

        pipeline = self.create_model_pipeline(
            maxIter=maxIter,
            maxDepth=maxDepth,
            stepSize=stepSize,
            seed=seed
        )
        model = pipeline.fit(train_df)

        if save_model:
            self.save_model(model, self.model_name, version)

        print("Gradient Boosted Trees training completed!")
        return model

    def get_feature_importance(self, model: PipelineModel) -> dict:
        """
        Extract feature importance from GBT model.

        Args:
            model: Trained PipelineModel

        Returns:
            Dictionary with feature names and importance scores (sorted)
        """
        gbt_model = model.stages[-1]  # Get GBT model from pipeline

        importances = gbt_model.featureImportances.toArray()
        feature_importance = {
            self.feature_cols[i]: float(importances[i])
            for i in range(len(self.feature_cols))
        }

        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

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

    spark = create_spark_session("GBTTraining")

    # Create pipeline
    gbt_pipeline = GBTPipeline(spark)

    # Load data
    df = gbt_pipeline.load_data()

    # Split data
    train_df, test_df = gbt_pipeline.train_test_split(df)

    # Train model
    model = gbt_pipeline.train(train_df, save_model=True)

    # Get feature importance
    importance = gbt_pipeline.get_feature_importance(model)
    print("\nTop 10 Features by Importance:")
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")

    # Evaluate
    metrics = gbt_pipeline.evaluate(model, test_df)

    spark.stop()

