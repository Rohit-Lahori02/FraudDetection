"""
Spark session utilities for fraud detection.
Creates optimized Spark sessions for EMR.
"""
from pyspark.sql import SparkSession

def create_spark_session(app_name="FraudDetection", master="yarn"):
    """
    Create optimized Spark session for EMR.
    
    Args:
        app_name: Name of the Spark application
        master: Spark master URL (default: "yarn" for EMR)
    
    Returns:
        Configured SparkSession instance
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

