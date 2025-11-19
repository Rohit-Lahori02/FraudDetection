"""
Comprehensive evaluation metrics for fraud detection models.
Handles imbalanced data and provides multiple performance measures.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationMetrics
import numpy as np
from typing import Dict, List, Tuple
import json
from src.common.config import config


class FraudDetectionMetrics:
    """Comprehensive metrics calculator for fraud detection."""

    def __init__(self, spark: SparkSession):
        """
        Initialize metrics calculator.

        Args:
            spark: SparkSession instance
        """
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

    def calculate_all_metrics(self, predictions_df: DataFrame) -> Dict[str, float]:
        """
        Calculate all evaluation metrics for fraud detection.

        Args:
            predictions_df: DataFrame with columns: label, prediction, rawPrediction (or probability)

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

    def _calculate_accuracy(self, predictions_df: DataFrame) -> float:
        """Calculate accuracy."""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_precision(self, predictions_df: DataFrame) -> float:
        """Calculate precision (positive predictive value)."""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_recall(self, predictions_df: DataFrame) -> float:
        """Calculate recall (sensitivity, true positive rate)."""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_f1_score(self, predictions_df: DataFrame) -> float:
        """Calculate F1 score."""
        return self.multiclass_evaluator.evaluate(predictions_df)

    def _calculate_auroc(self, predictions_df: DataFrame) -> float:
        """Calculate Area Under ROC Curve."""
        return self.binary_evaluator.evaluate(predictions_df)

    def _calculate_auprc(self, predictions_df: DataFrame) -> float:
        """Calculate Area Under Precision-Recall Curve."""
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )
        return evaluator.evaluate(predictions_df)

    def _calculate_confusion_matrix(self, predictions_df: DataFrame) -> Dict[str, int]:
        """
        Calculate confusion matrix components.
        
        Args:
            predictions_df: DataFrame with label and prediction columns
            
        Returns:
            Dictionary with tp, tn, fp, fn counts
        """
        # Use Spark aggregations for scalability
        # Calculate TP, TN, FP, FN using conditional aggregations
        result = predictions_df.select(
            F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
            F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn")
        ).collect()[0]

        return {
            "tp": int(result["tp"]) if result["tp"] else 0,
            "tn": int(result["tn"]) if result["tn"] else 0,
            "fp": int(result["fp"]) if result["fp"] else 0,
            "fn": int(result["fn"]) if result["fn"] else 0
        }

    def _calculate_specificity(self, confusion_matrix: Dict[str, int]) -> float:
        """
        Calculate specificity (true negative rate).

        Args:
            confusion_matrix: Dictionary with tp, tn, fp, fn

        Returns:
            Specificity score
        """
        tn = confusion_matrix["tn"]
        fp = confusion_matrix["fp"]
        if (tn + fp) == 0:
            return 0.0
        return tn / (tn + fp)

    def save_metrics_to_s3(self, metrics: Dict, model_name: str, version: str = "v1"):
        """
        Save metrics to S3 as JSON.

        Args:
            metrics: Dictionary with metrics to save
            model_name: Name of the model
            version: Model version (default: "v1")
        """
        import boto3
        from botocore.exceptions import ClientError

        s3_key = f"outputs/evaluation/{model_name}_{version}_metrics.json"
        
        try:
            s3_client = boto3.client('s3')
            # Handle both s3://bucket/key and bucket/key formats
            if s3_key.startswith("s3://"):
                bucket, key = s3_key.replace("s3://", "").split("/", 1)
            elif s3_key.startswith(config.OUTPUTS_PATH):
                # If s3_key includes full path from config
                full_path = f"{config.OUTPUTS_PATH}evaluation/{model_name}_{version}_metrics.json"
                if full_path.startswith("s3://"):
                    bucket, key = full_path.replace("s3://", "").split("/", 1)
                else:
                    bucket = config.S3_BUCKET
                    key = s3_key
            else:
                bucket = config.S3_BUCKET
                key = s3_key

            metrics_json = json.dumps(metrics, indent=2, default=str)
            
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=metrics_json,
                ContentType="application/json"
            )
            print(f"Metrics saved to s3://{bucket}/{key}")
        except ClientError as e:
            print(f"Error saving metrics to S3: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            print(f"Note: S3 operation requires AWS credentials. Metrics saved locally.")
            # Save locally as fallback
            from datetime import datetime
            local_path = f"{model_name}_{version}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(local_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"Metrics saved locally to {local_path}")

    @staticmethod
    def load_metrics_from_s3(s3_path: str) -> Dict:
        """
        Load metrics from S3.

        Args:
            s3_path: S3 path to metrics JSON

        Returns:
            Dictionary with metrics
        """
        import boto3
        import json
        from botocore.exceptions import ClientError

        try:
            s3_client = boto3.client('s3')
            # Handle both s3://bucket/key and bucket/key formats
            if s3_path.startswith("s3://"):
                bucket, key = s3_path.replace("s3://", "").split("/", 1)
            else:
                bucket, key = s3_path.split("/", 1)

            response = s3_client.get_object(Bucket=bucket, Key=key)
            metrics = json.loads(response['Body'].read().decode('utf-8'))
            print(f"Metrics loaded from {s3_path}")
            return metrics
        except ClientError as e:
            print(f"Error loading metrics from S3: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise


# Usage example
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session
    from pyspark.ml import PipelineModel

    spark = create_spark_session("MetricsEvaluation")

    # Example: Load model and test data (from Rohit's pipeline)
    # model = PipelineModel.load("s3://bucket/models/logistic_regression/v1/model/")
    # test_df = spark.read.parquet("s3://bucket/processed/test/")
    # predictions = model.transform(test_df)

    # For testing with mock data
    print("=== Metrics Calculator Initialized ===")
    print("Use calculate_all_metrics(predictions_df) to evaluate model predictions")
    print("predictions_df must have columns: label, prediction, rawPrediction")

    # Example usage (commented out for testing):
    # metrics_calculator = FraudDetectionMetrics(spark)
    # metrics = metrics_calculator.calculate_all_metrics(predictions)
    #
    # print("\n=== Evaluation Metrics ===")
    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"F1 Score: {metrics['f1_score']:.4f}")
    # print(f"AUROC: {metrics['auroc']:.4f}")
    # print(f"AUPRC: {metrics['auprc']:.4f}")
    #
    # metrics_calculator.save_metrics_to_s3(metrics, "logistic_regression", "v1")

    spark.stop()

