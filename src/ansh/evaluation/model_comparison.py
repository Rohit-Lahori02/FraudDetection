"""
Model comparison framework with statistical significance testing.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from src.ansh.evaluation.metrics import FraudDetectionMetrics
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy import stats
from src.common.config import config


class ModelComparator:
    """Compare multiple models with statistical testing."""

    def __init__(self, spark: SparkSession):
        """
        Initialize model comparator.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        self.metrics_calculator = FraudDetectionMetrics(spark)
        self.results = []

    def compare_models(
        self,
        models: Dict[str, PipelineModel],
        test_df: DataFrame,
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
            if model_name not in models:
                print(f"Warning: Model {model_name} not found in models dictionary")
                continue

            model = models[model_name]

            # Make predictions
            predictions = model.transform(test_df)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(predictions)
            metrics["model_name"] = model_name

            comparison_results.append(metrics)
            self.results.append(metrics)

        if not comparison_results:
            return pd.DataFrame()

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)

        # Sort by AUROC (descending)
        comparison_df = comparison_df.sort_values("auroc", ascending=False)

        return comparison_df

    def compare_metrics(
        self,
        metrics_list: List[Dict],
        model_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare models using pre-calculated metrics.

        Args:
            metrics_list: List of metric dictionaries
            model_names: Optional list of model names

        Returns:
            DataFrame with comparison results
        """
        if model_names:
            for i, name in enumerate(model_names):
                if i < len(metrics_list):
                    metrics_list[i]["model_name"] = name
        else:
            for i, metrics in enumerate(metrics_list):
                if "model_name" not in metrics:
                    metrics["model_name"] = f"model_{i+1}"

        comparison_df = pd.DataFrame(metrics_list)
        comparison_df = comparison_df.sort_values("auroc", ascending=False)

        return comparison_df

    def statistical_significance_test(
        self,
        model1_predictions: DataFrame,
        model2_predictions: DataFrame,
        metric: str = "auroc"
    ) -> Dict:
        """
        Perform statistical significance test between two models using bootstrap.

        Args:
            model1_predictions: Predictions DataFrame from model 1
            model2_predictions: Predictions DataFrame from model 2
            metric: Metric to compare (default: "auroc")

        Returns:
            Dictionary with test results
        """
        n_bootstrap = 100  # Reduced for performance, can increase if needed
        model1_scores = []
        model2_scores = []

        # Bootstrap sampling
        for i in range(n_bootstrap):
            # Sample with replacement
            sample1 = model1_predictions.sample(
                True, 1.0, seed=np.random.randint(0, 10000) + i
            )
            sample2 = model2_predictions.sample(
                True, 1.0, seed=np.random.randint(0, 10000) + i
            )

            metrics1 = self.metrics_calculator.calculate_all_metrics(sample1)
            metrics2 = self.metrics_calculator.calculate_all_metrics(sample2)

            model1_scores.append(metrics1.get(metric, 0.0))
            model2_scores.append(metrics2.get(metric, 0.0))

        # Perform paired t-test
        differences = np.array(model1_scores) - np.array(model2_scores)
        t_stat, p_value = stats.ttest_1samp(differences, 0)

        result = {
            "model1_mean": float(np.mean(model1_scores)),
            "model2_mean": float(np.mean(model2_scores)),
            "difference": float(np.mean(differences)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        }

        return result

    def generate_comparison_report(self, comparison_df: pd.DataFrame) -> str:
        """
        Generate formatted comparison report.

        Args:
            comparison_df: DataFrame with comparison results

        Returns:
            Formatted report string
        """
        if comparison_df.empty:
            return "No comparison results available."

        report = "\n=== Model Comparison Report ===\n\n"
        
        # Select key metrics for display
        display_cols = ["model_name", "auroc", "auprc", "precision", "recall", "f1_score", "balanced_accuracy"]
        available_cols = [col for col in display_cols if col in comparison_df.columns]
        
        report += comparison_df[available_cols].to_string(index=False)
        report += "\n\n"

        # Best model
        best_model = comparison_df.iloc[0]
        report += f"Best Model: {best_model['model_name']}\n"
        if 'auroc' in comparison_df.columns:
            report += f"  AUROC: {best_model['auroc']:.4f}\n"
        if 'auprc' in comparison_df.columns:
            report += f"  AUPRC: {best_model['auprc']:.4f}\n"
        if 'f1_score' in comparison_df.columns:
            report += f"  F1 Score: {best_model['f1_score']:.4f}\n"

        return report

    def save_report_to_s3(
        self,
        comparison_df: pd.DataFrame,
        report_name: str = "model_comparison",
        format: str = "json"
    ):
        """
        Save comparison report to S3.

        Args:
            comparison_df: DataFrame with comparison results
            report_name: Name for the report file
            format: Format to save ("json" or "csv")
        """
        import boto3
        import json
        from datetime import datetime
        from botocore.exceptions import ClientError

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"outputs/comparison/{report_name}_{timestamp}.{format}"

        try:
            s3_client = boto3.client('s3')
            
            if s3_key.startswith("s3://"):
                bucket, key = s3_key.replace("s3://", "").split("/", 1)
            else:
                bucket = config.S3_BUCKET
                key = s3_key

            if format == "json":
                body = comparison_df.to_json(orient='records', indent=2)
                content_type = "application/json"
            else:  # csv
                body = comparison_df.to_csv(index=False)
                content_type = "text/csv"

            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=content_type
            )
            print(f"Comparison report saved to s3://{bucket}/{key}")
        except ClientError as e:
            print(f"Error saving comparison report to S3: {e}")
            # Save locally as fallback
            local_path = f"{report_name}_{timestamp}.{format}"
            if format == "json":
                comparison_df.to_json(local_path, orient='records', indent=2)
            else:
                comparison_df.to_csv(local_path, index=False)
            print(f"Comparison report saved locally to {local_path}")


# Usage example
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session
    from src.common.config import config

    spark = create_spark_session("ModelComparison")

    # Example: Load models (when available in S3)
    # models = {
    #     "logistic_regression": PipelineModel.load(f"{config.MODELS_PATH}logistic_regression/v1/model/"),
    #     "random_forest": PipelineModel.load(f"{config.MODELS_PATH}random_forest/v1/model/"),
    #     "gbt_classifier": PipelineModel.load(f"{config.MODELS_PATH}gbt_classifier/v1/model/")
    # }
    #
    # # Load test data
    # test_df = spark.read.parquet(f"{config.PROCESSED_DATA_PATH}test/")
    #
    # # Compare models
    # comparator = ModelComparator(spark)
    # comparison_df = comparator.compare_models(models, test_df)
    #
    # # Print report
    # print(comparator.generate_comparison_report(comparison_df))
    #
    # # Save comparison
    # comparator.save_report_to_s3(comparison_df)

    print("Model comparison module ready!")

    spark.stop()

