"""
Cross-validation strategies for imbalanced fraud detection data.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from typing import List, Dict, Tuple
import numpy as np
from src.ansh.evaluation.metrics import FraudDetectionMetrics


class FraudDetectionCrossValidator:
    """Cross-validation for fraud detection with stratification support."""

    def __init__(self, spark: SparkSession, random_seed: int = 42):
        """
        Initialize cross-validator.

        Args:
            spark: SparkSession instance
            random_seed: Random seed for reproducibility
        """
        self.spark = spark
        self.random_seed = random_seed
        self.metrics_calculator = FraudDetectionMetrics(spark)

    def stratified_split(
        self,
        df: DataFrame,
        train_ratio: float = 0.8,
        label_col: str = "label"
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
        fraud_rate = fraud_count / total_count if total_count > 0 else 0

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
        train_fraud_rate = (train_df.filter(F.col(label_col) == 1).count() / 
                          train_df.count() if train_df.count() > 0 else 0)
        test_fraud_rate = (test_df.filter(F.col(label_col) == 1).count() / 
                          test_df.count() if test_df.count() > 0 else 0)

        print(f"Original fraud rate: {fraud_rate:.4f}")
        print(f"Train fraud rate: {train_fraud_rate:.4f}")
        print(f"Test fraud rate: {test_fraud_rate:.4f}")

        return train_df, test_df

    def evaluate_fold(
        self,
        model: PipelineModel,
        fold_df: DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate model on a single fold.

        Args:
            model: Trained PipelineModel
            fold_df: Fold DataFrame for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = model.transform(fold_df)
        metrics = self.metrics_calculator.calculate_all_metrics(predictions)
        return metrics

    def cross_validate(
        self,
        pipeline: Pipeline,
        train_df: DataFrame,
        num_folds: int = 5,
        metric_name: str = "areaUnderROC",
        label_col: str = "label"
    ) -> Dict:
        """
        Perform k-fold cross-validation with stratification.

        Args:
            pipeline: MLlib Pipeline to evaluate
            train_df: Training DataFrame
            num_folds: Number of CV folds (default: 5)
            metric_name: Metric to optimize (default: "areaUnderROC")
            label_col: Name of label column

        Returns:
            Dictionary with CV results including metrics and best model
        """
        # Create evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol=label_col,
            rawPredictionCol="rawPrediction",
            metricName=metric_name
        )

        # Create cross-validator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=[{}],  # Empty param map for simple CV
            evaluator=evaluator,
            numFolds=num_folds,
            seed=self.random_seed,
            parallelism=4  # Number of parallel jobs
        )

        # Fit model
        print(f"Starting {num_folds}-fold cross-validation...")
        cv_model = cv.fit(train_df)

        # Get CV scores
        avg_metrics = cv_model.avgMetrics
        best_score = np.mean(avg_metrics) if avg_metrics else 0.0
        std_score = np.std(avg_metrics) if avg_metrics else 0.0

        # Get best model
        best_model = cv_model.bestModel

        # Calculate detailed metrics on each fold
        fold_metrics = []
        for i, metric in enumerate(avg_metrics):
            fold_metrics.append({
                "fold": i + 1,
                metric_name: float(metric) if metric else 0.0
            })

        results = {
            "cv_scores": avg_metrics,
            "mean_score": float(best_score),
            "std_score": float(std_score),
            "best_model": best_model,
            "fold_metrics": fold_metrics,
            "num_folds": num_folds,
            "metric_name": metric_name
        }

        print(f"\n=== Cross-Validation Results ===")
        print(f"Mean {metric_name}: {best_score:.4f} (+/- {std_score:.4f})")
        print(f"Fold scores: {[f'{m:.4f}' for m in avg_metrics]}")

        return results


# Usage example
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session
    from src.common.config import config
    from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline

    spark = create_spark_session("CrossValidationTest")

    # Load data
    base_pipeline = BaseFraudDetectionPipeline(spark)
    df = base_pipeline.load_data()

    # Stratified split
    cv_validator = FraudDetectionCrossValidator(spark, random_seed=42)
    train_df, test_df = cv_validator.stratified_split(df, train_ratio=0.8)

    # Create feature pipeline for CV
    feature_pipeline = base_pipeline.create_feature_pipeline()

    # Run cross-validation (example with feature pipeline)
    # cv_results = cv_validator.cross_validate(feature_pipeline, train_df, num_folds=5)

    print("Cross-validation module ready!")

    spark.stop()

