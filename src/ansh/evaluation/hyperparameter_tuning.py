"""
Hyperparameter tuning with MLflow tracking.
"""
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from typing import Dict, Tuple
import numpy as np
from src.ansh.evaluation.metrics import FraudDetectionMetrics
from src.rohit.pipelines.logistic_regression import LogisticRegressionPipeline
from src.rohit.pipelines.random_forest import RandomForestPipeline
from src.rohit.pipelines.gbt_classifier import GBTPipeline


class HyperparameterTuner:
    """Hyperparameter tuning with MLflow integration."""

    def __init__(self, spark: SparkSession, experiment_name: str = "fraud_detection_tuning"):
        """
        Initialize hyperparameter tuner.

        Args:
            spark: SparkSession instance
            experiment_name: MLflow experiment name
        """
        self.spark = spark
        self.experiment_name = experiment_name
        self.metrics_calculator = FraudDetectionMetrics(spark)
        
        # Set up MLflow tracking
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set MLflow experiment: {e}")
            print("Continuing without MLflow tracking...")

    def tune_logistic_regression(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        num_folds: int = 5
    ) -> Tuple[Pipeline, Dict]:
        """
        Tune Logistic Regression hyperparameters.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            num_folds: Number of CV folds (default: 5)

        Returns:
            Tuple of (best_model, best_params_dict)
        """
        print("Tuning Logistic Regression hyperparameters...")

        # Create pipeline
        lr_pipeline_class = LogisticRegressionPipeline(self.spark)
        base_pipeline = lr_pipeline_class.create_model_pipeline()

        # Get the LR model from pipeline
        lr_model = base_pipeline.getStages()[-1]

        # Define parameter grid
        param_grid = ParamGridBuilder() \
            .addGrid(lr_model.regParam, [0.01, 0.1, 1.0]) \
            .addGrid(lr_model.elasticNetParam, [0.0, 0.5, 1.0]) \
            .addGrid(lr_model.maxIter, [100, 200]) \
            .build()

        # Create evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        # Create cross-validator
        cv = CrossValidator(
            estimator=base_pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42,
            parallelism=4
        )

        # Run tuning
        with mlflow.start_run(run_name="LR_Hyperparameter_Tuning"):
            print("Running cross-validation...")
            cv_model = cv.fit(train_df)

            # Get best model
            best_model = cv_model.bestModel
            best_params = best_model.stages[-1].extractParamMap()
            cv_scores = cv_model.avgMetrics

            # Evaluate on test set
            test_predictions = best_model.transform(test_df)
            test_metrics = self.metrics_calculator.calculate_all_metrics(test_predictions)

            # Log to MLflow
            try:
                for param, value in best_params.items():
                    mlflow.log_param(str(param.name), value)
                
                mlflow.log_metrics({
                    "test_auroc": test_metrics["auroc"],
                    "test_auprc": test_metrics["auprc"],
                    "test_f1": test_metrics["f1_score"],
                    "cv_mean_auroc": float(np.mean(cv_scores)),
                    "cv_std_auroc": float(np.std(cv_scores))
                })

                # Log model
                mlflow.spark.log_model(best_model, "model")
            except Exception as e:
                print(f"Warning: Could not log to MLflow: {e}")

            print(f"\n=== LR Tuning Results ===")
            print(f"Best test AUROC: {test_metrics['auroc']:.4f}")
            print(f"Best test AUPRC: {test_metrics['auprc']:.4f}")
            print(f"CV Mean AUROC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

            # Convert best_params to dict
            best_params_dict = {str(param.name): value for param, value in best_params.items()}

            return best_model, best_params_dict

    def tune_random_forest(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        num_folds: int = 5
    ) -> Tuple[Pipeline, Dict]:
        """
        Tune Random Forest hyperparameters.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            num_folds: Number of CV folds (default: 5)

        Returns:
            Tuple of (best_model, best_params_dict)
        """
        print("Tuning Random Forest hyperparameters...")

        rf_pipeline_class = RandomForestPipeline(self.spark)
        base_pipeline = rf_pipeline_class.create_model_pipeline()

        rf_model = base_pipeline.getStages()[-1]

        # Define parameter grid
        param_grid = ParamGridBuilder() \
            .addGrid(rf_model.numTrees, [50, 100, 200]) \
            .addGrid(rf_model.maxDepth, [5, 10, 15]) \
            .addGrid(rf_model.impurity, ["gini", "entropy"]) \
            .build()

        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        cv = CrossValidator(
            estimator=base_pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42,
            parallelism=4
        )

        with mlflow.start_run(run_name="RF_Hyperparameter_Tuning"):
            print("Running cross-validation...")
            cv_model = cv.fit(train_df)

            best_model = cv_model.bestModel
            best_params = best_model.stages[-1].extractParamMap()
            cv_scores = cv_model.avgMetrics

            test_predictions = best_model.transform(test_df)
            test_metrics = self.metrics_calculator.calculate_all_metrics(test_predictions)

            try:
                for param, value in best_params.items():
                    mlflow.log_param(str(param.name), value)
                
                mlflow.log_metrics({
                    "test_auroc": test_metrics["auroc"],
                    "test_auprc": test_metrics["auprc"],
                    "test_f1": test_metrics["f1_score"],
                    "cv_mean_auroc": float(np.mean(cv_scores)),
                    "cv_std_auroc": float(np.std(cv_scores))
                })

                mlflow.spark.log_model(best_model, "model")
            except Exception as e:
                print(f"Warning: Could not log to MLflow: {e}")

            print(f"\n=== RF Tuning Results ===")
            print(f"Best test AUROC: {test_metrics['auroc']:.4f}")
            print(f"Best test AUPRC: {test_metrics['auprc']:.4f}")
            print(f"CV Mean AUROC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

            best_params_dict = {str(param.name): value for param, value in best_params.items()}

            return best_model, best_params_dict

    def tune_gbt_classifier(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        num_folds: int = 5
    ) -> Tuple[Pipeline, Dict]:
        """
        Tune GBT Classifier hyperparameters.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            num_folds: Number of CV folds (default: 5)

        Returns:
            Tuple of (best_model, best_params_dict)
        """
        print("Tuning GBT Classifier hyperparameters...")

        gbt_pipeline_class = GBTPipeline(self.spark)
        base_pipeline = gbt_pipeline_class.create_model_pipeline()

        gbt_model = base_pipeline.getStages()[-1]

        # Define parameter grid
        param_grid = ParamGridBuilder() \
            .addGrid(gbt_model.maxIter, [50, 100, 150]) \
            .addGrid(gbt_model.maxDepth, [3, 5, 7]) \
            .addGrid(gbt_model.stepSize, [0.05, 0.1, 0.15]) \
            .build()

        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        cv = CrossValidator(
            estimator=base_pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42,
            parallelism=4
        )

        with mlflow.start_run(run_name="GBT_Hyperparameter_Tuning"):
            print("Running cross-validation...")
            cv_model = cv.fit(train_df)

            best_model = cv_model.bestModel
            best_params = best_model.stages[-1].extractParamMap()
            cv_scores = cv_model.avgMetrics

            test_predictions = best_model.transform(test_df)
            test_metrics = self.metrics_calculator.calculate_all_metrics(test_predictions)

            try:
                for param, value in best_params.items():
                    mlflow.log_param(str(param.name), value)
                
                mlflow.log_metrics({
                    "test_auroc": test_metrics["auroc"],
                    "test_auprc": test_metrics["auprc"],
                    "test_f1": test_metrics["f1_score"],
                    "cv_mean_auroc": float(np.mean(cv_scores)),
                    "cv_std_auroc": float(np.std(cv_scores))
                })

                mlflow.spark.log_model(best_model, "model")
            except Exception as e:
                print(f"Warning: Could not log to MLflow: {e}")

            print(f"\n=== GBT Tuning Results ===")
            print(f"Best test AUROC: {test_metrics['auroc']:.4f}")
            print(f"Best test AUPRC: {test_metrics['auprc']:.4f}")
            print(f"CV Mean AUROC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

            best_params_dict = {str(param.name): value for param, value in best_params.items()}

            return best_model, best_params_dict


# Usage example
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session
    from src.common.config import config
    from src.rohit.pipelines.base_pipeline import BaseFraudDetectionPipeline

    spark = create_spark_session("HyperparameterTuning")

    # Load data
    base_pipeline = BaseFraudDetectionPipeline(spark)
    df = base_pipeline.load_data()

    # Split data
    train_df, test_df = base_pipeline.train_test_split(df)

    # Tune models
    tuner = HyperparameterTuner(spark)

    # Example tuning (commented out as it takes time)
    # lr_model, lr_params = tuner.tune_logistic_regression(train_df, test_df)
    # rf_model, rf_params = tuner.tune_random_forest(train_df, test_df)
    # gbt_model, gbt_params = tuner.tune_gbt_classifier(train_df, test_df)

    print("Hyperparameter tuning module ready!")

    spark.stop()

