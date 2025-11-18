"""
Comprehensive data profiling for credit card fraud detection dataset.
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
from src.common.config import config
from src.common.schema_validator import get_credit_card_schema


class CreditCardDataProfiler:
    """Data profiler for credit card fraud detection."""

    def __init__(self, spark: SparkSession):
        """
        Initialize data profiler.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark

    def profile_dataset(self, df: DataFrame) -> Dict:
        """
        Generate comprehensive data profile.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with profiling results
        """
        profile = {
            "basic_statistics": self._basic_statistics(df),
            "missing_values": self._missing_values(df),
            "fraud_statistics": self._fraud_statistics(df),
            "correlation_analysis": self._correlation_analysis(df),
            "distribution_statistics": self._distribution_statistics(df),
            "data_quality": self._data_quality_checks(df)
        }

        return profile

    def _basic_statistics(self, df: DataFrame) -> Dict:
        """Calculate basic statistics."""
        total_count = df.count()
        num_features = len(df.columns)

        stats = {
            "total_records": total_count,
            "num_features": num_features,
            "feature_names": df.columns
        }

        return stats

    def _missing_values(self, df: DataFrame) -> Dict:
        """Detect missing values."""
        missing_counts = {}
        total_count = df.count()

        for col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            missing_counts[col] = {
                "count": null_count,
                "percentage": (null_count / total_count) * 100 if total_count > 0 else 0
            }

        return missing_counts

    def _fraud_statistics(self, df: DataFrame) -> Dict:
        """Calculate fraud-related statistics."""
        total_count = df.count()
        fraud_count = df.filter(F.col("Class") == 1).count()
        normal_count = df.filter(F.col("Class") == 0).count()

        fraud_rate = (fraud_count / total_count) * 100 if total_count > 0 else 0

        # Amount statistics by class
        fraud_amount_stats = df.filter(F.col("Class") == 1).agg(
            F.avg("Amount").alias("avg_amount"),
            F.stddev("Amount").alias("std_amount"),
            F.min("Amount").alias("min_amount"),
            F.max("Amount").alias("max_amount")
        ).collect()[0]

        normal_amount_stats = df.filter(F.col("Class") == 0).agg(
            F.avg("Amount").alias("avg_amount"),
            F.stddev("Amount").alias("std_amount"),
            F.min("Amount").alias("min_amount"),
            F.max("Amount").alias("max_amount")
        ).collect()[0]

        stats = {
            "total_transactions": total_count,
            "fraud_transactions": fraud_count,
            "normal_transactions": normal_count,
            "fraud_rate_percentage": fraud_rate,
            "fraud_amount_stats": {
                "avg": float(fraud_amount_stats["avg_amount"]) if fraud_amount_stats["avg_amount"] else 0.0,
                "std": float(fraud_amount_stats["std_amount"]) if fraud_amount_stats["std_amount"] else 0.0,
                "min": float(fraud_amount_stats["min_amount"]) if fraud_amount_stats["min_amount"] else 0.0,
                "max": float(fraud_amount_stats["max_amount"]) if fraud_amount_stats["max_amount"] else 0.0
            },
            "normal_amount_stats": {
                "avg": float(normal_amount_stats["avg_amount"]) if normal_amount_stats["avg_amount"] else 0.0,
                "std": float(normal_amount_stats["std_amount"]) if normal_amount_stats["std_amount"] else 0.0,
                "min": float(normal_amount_stats["min_amount"]) if normal_amount_stats["min_amount"] else 0.0,
                "max": float(normal_amount_stats["max_amount"]) if normal_amount_stats["max_amount"] else 0.0
            }
        }

        return stats

    def _correlation_analysis(self, df: DataFrame) -> Dict:
        """Perform correlation analysis."""
        # Select numeric columns - using 'id' instead of 'Time' as per actual dataset schema
        numeric_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        if "id" in df.columns:
            numeric_cols.append("id")

        # Calculate correlation with Class
        correlations = {}
        for col in numeric_cols:
            if col in df.columns:
                try:
                    corr = df.stat.corr(col, "Class")
                    correlations[col] = float(corr) if corr is not None else 0.0
                except:
                    correlations[col] = 0.0

        # Sort by absolute correlation
        sorted_correlations = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "class_correlations": sorted_correlations,
            "top_correlated_features": list(sorted_correlations.keys())[:10]
        }

    def _distribution_statistics(self, df: DataFrame) -> Dict:
        """Calculate distribution statistics."""
        # Using 'id' instead of 'Time' as per actual dataset schema
        numeric_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        if "id" in df.columns:
            numeric_cols.append("id")

        distribution_stats = {}
        # Sample first 5 numeric columns for performance
        for col in numeric_cols[:5]:
            if col in df.columns:
                stats = df.select(
                    F.mean(col).alias("mean"),
                    F.stddev(col).alias("std"),
                    F.min(col).alias("min"),
                    F.max(col).alias("max"),
                    F.percentile_approx(col, 0.25).alias("q1"),
                    F.percentile_approx(col, 0.5).alias("median"),
                    F.percentile_approx(col, 0.75).alias("q3")
                ).collect()[0]

                distribution_stats[col] = {
                    "mean": float(stats["mean"]) if stats["mean"] else 0.0,
                    "std": float(stats["std"]) if stats["std"] else 0.0,
                    "min": float(stats["min"]) if stats["min"] else 0.0,
                    "max": float(stats["max"]) if stats["max"] else 0.0,
                    "q1": float(stats["q1"]) if stats["q1"] else 0.0,
                    "median": float(stats["median"]) if stats["median"] else 0.0,
                    "q3": float(stats["q3"]) if stats["q3"] else 0.0
                }

        return distribution_stats

    def _data_quality_checks(self, df: DataFrame) -> Dict:
        """Perform data quality checks."""
        checks = {
            "has_missing_values": False,
            "has_duplicates": False,
            "has_negative_amounts": False,
            "has_invalid_class_labels": False,
            "data_quality_passed": True
        }

        # Check missing values
        for col in df.columns:
            if df.filter(F.col(col).isNull()).count() > 0:
                checks["has_missing_values"] = True
                checks["data_quality_passed"] = False
                break

        # Check duplicates (using all columns)
        duplicate_count = df.groupBy(df.columns).count().filter(F.col("count") > 1).count()
        if duplicate_count > 0:
            checks["has_duplicates"] = True

        # Check negative amounts
        if "Amount" in df.columns:
            negative_amounts = df.filter(F.col("Amount") < 0).count()
            if negative_amounts > 0:
                checks["has_negative_amounts"] = True

        # Check class labels (should be 0 or 1)
        if "Class" in df.columns:
            invalid_labels = df.filter(~F.col("Class").isin([0, 1])).count()
            if invalid_labels > 0:
                checks["has_invalid_class_labels"] = True
                checks["data_quality_passed"] = False

        return checks

    def save_profile_to_s3(self, profile: Dict, output_path: str = None):
        """
        Save profiling results to S3.

        Args:
            profile: Profile dictionary to save
            output_path: S3 path (optional, uses default if None)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = f"s3://{config.S3_BUCKET}/profiling/metadata_{timestamp}.json"
        
        import boto3
        from botocore.exceptions import ClientError

        try:
            s3_client = boto3.client('s3')
            # Handle both s3://bucket/key and bucket/key formats
            if output_path.startswith("s3://"):
                bucket, key = output_path.replace("s3://", "").split("/", 1)
            else:
                bucket, key = output_path.split("/", 1)

            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(profile, indent=2, default=str),
                ContentType="application/json"
            )

            print(f"Profile saved to {output_path}")
        except ClientError as e:
            print(f"Error saving profile to S3: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            print(f"Note: S3 operation requires AWS credentials. Profile saved locally.")
            # Save locally as fallback
            local_path = key.split("/")[-1] if "/" in key else f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(local_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            print(f"Profile saved locally to {local_path}")

    @staticmethod
    def get_profiling_metadata(s3_path: str) -> Dict:
        """
        Load profiling metadata from S3.

        Args:
            s3_path: S3 path to profiling metadata JSON

        Returns:
            Dictionary with profiling metadata
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
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            print(f"Profiling metadata loaded from {s3_path}")
            return metadata
        except ClientError as e:
            print(f"Error loading profiling metadata from S3: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise


# Usage
if __name__ == "__main__":
    from src.common.spark_session import create_spark_session

    spark = create_spark_session("DataProfiling")

    # Load data
    df = spark.read.csv(config.RAW_DATA_PATH, header=True, schema=get_credit_card_schema())

    # Profile data
    profiler = CreditCardDataProfiler(spark)
    profile = profiler.profile_dataset(df)

    # Print summary
    print("\n=== Data Profile Summary ===")
    print(f"Total Records: {profile['basic_statistics']['total_records']:,}")
    print(f"Fraud Rate: {profile['fraud_statistics']['fraud_rate_percentage']:.2f}%")
    print(f"Data Quality Passed: {profile['data_quality']['data_quality_passed']}")

    # Save to S3
    profiler.save_profile_to_s3(profile)

    spark.stop()

