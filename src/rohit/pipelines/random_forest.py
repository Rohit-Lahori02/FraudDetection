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
        """
        Initialize Random Forest pipeline.

        Args:
            spark: SparkSession instance
        """
        super().__init__(spark)
        self.model_name = "random_forest"

    def create_model_pipeline(
        self,
        numTrees: int = 100,
        maxDepth: int = 10,
        impurity: str = "gini",
        seed: int = 42
    ) -> Pipeline:
        """
        Create complete pipeline with Random Forest.

        Args:
            numTrees: Number of trees (default: 100)
            maxDepth: Maximum depth of trees (default: 10)
            impurity: Impurity criterion ("gini" or "entropy", default: "gini")
            seed: Random seed (default: 42)

        Returns:
            Pipeline with feature engineering and classifier
        """
        feature_pipeline = self.create_feature_pipeline()

        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol="label",
            numTrees=numTrees,
            maxDepth=maxDepth,
            impurity=impurity,
            seed=seed
        )

        pipeline = Pipeline(stages=feature_pipeline.getStages() + [rf])
        return pipeline

    def train(
        self,
        train_df: DataFrame,
        save_model: bool = True,
        version: str = "v1",
        numTrees: int = 100,
        maxDepth: int = 10,
        impurity: str = "gini",
        seed: int = 42
    ) -> PipelineModel:
        """
        Train Random Forest model.

        Args:
            train_df: Training DataFrame
            save_model: Whether to save model to S3
            version: Model version (default: "v1")
            numTrees: Number of trees
            maxDepth: Maximum depth of trees
            impurity: Impurity criterion
            seed: Random seed

        Returns:
            Trained PipelineModel
        """
        print("Training Random Forest model...")

        pipeline = self.create_model_pipeline(
            numTrees=numTrees,
            maxDepth=maxDepth,
            impurity=impurity,
            seed=seed
        )
        model = pipeline.fit(train_df)

        if save_model:
            self.save_model(model, self.model_name, version)

        print("Random Forest training completed!")
        return model

    def get_feature_importance(self, model: PipelineModel) -> dict:
        """
        Extract feature importance from Random Forest model.

        Args:
            model: Trained PipelineModel

        Returns:
            Dictionary with feature names and importance scores (sorted)
        """
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

    spark = create_spark_session("RandomForestTraining")

    # Create pipeline
    rf_pipeline = RandomForestPipeline(spark)

    # Load data
    df = rf_pipeline.load_data()

    # Split data
    train_df, test_df = rf_pipeline.train_test_split(df)

    # Train model
    model = rf_pipeline.train(train_df, save_model=True)

    # Get feature importance
    importance = rf_pipeline.get_feature_importance(model)
    print("\nTop 10 Features by Importance:")
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")

    # Evaluate
    metrics = rf_pipeline.evaluate(model, test_df)

    spark.stop()

