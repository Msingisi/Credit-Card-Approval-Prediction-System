from pydantic import BaseModel
from typing import Literal

class ModelNameConfig(BaseModel):
    """Model configuration for training"""
    model_name: Literal["RandomForest", "GradientBoosting", "XGBoost", "MLP", "LogisticRegression"]
