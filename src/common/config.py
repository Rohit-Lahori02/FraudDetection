"""
Configuration module for fraud detection project.
Handles S3 bucket and path configurations.
"""
import os
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    """Project configuration with S3 paths and settings."""
    S3_BUCKET: str = os.getenv("S3_BUCKET", "csp554-fraud-detection-default")
    RAW_DATA_PATH: str = None
    PROCESSED_DATA_PATH: str = None
    MODELS_PATH: str = None
    OUTPUTS_PATH: str = None

    def __post_init__(self):
        """Initialize S3 paths after instance creation."""
        if self.RAW_DATA_PATH is None:
            self.RAW_DATA_PATH = f"s3://{self.S3_BUCKET}/raw-data/creditcard_2023.csv"
        if self.PROCESSED_DATA_PATH is None:
            self.PROCESSED_DATA_PATH = f"s3://{self.S3_BUCKET}/processed/"
        if self.MODELS_PATH is None:
            self.MODELS_PATH = f"s3://{self.S3_BUCKET}/models/"
        if self.OUTPUTS_PATH is None:
            self.OUTPUTS_PATH = f"s3://{self.S3_BUCKET}/outputs/"

    @classmethod
    def from_env(cls):
        """
        Create ProjectConfig instance from environment variables.
        
        Returns:
            ProjectConfig instance with S3_BUCKET from environment or default
        """
        return cls(S3_BUCKET=os.getenv("S3_BUCKET", "csp554-fraud-detection-default"))

# Global config instance
config = ProjectConfig.from_env()

