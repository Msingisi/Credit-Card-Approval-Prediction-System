import logging
import pandas as pd
import mlflow
from zenml import step
from .config import ModelNameConfig
from src.model_dev import (RandomForestModel, MLPModel, GradientBoostingModel, LogisticRegressionModel,
                           XGBoostModel,)
from .config import ModelNameConfig
from zenml.client import Client
from sklearn.base import ClassifierMixin

# Get experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> ClassifierMixin:
    try:
        model_map = {
            "RandomForest": RandomForestModel(),
            "MLP": MLPModel(),
            "GradientBoosting": GradientBoostingModel(),
            "LogisticRegression": LogisticRegressionModel(),
            "XGBoost": XGBoostModel(),
        }

        if config.model_name not in model_map:
            raise ValueError(f"Unsupported model: {config.model_name}")
        mlflow.sklearn.autolog()
        model = model_map[config.model_name]
        trained_model = model.train(X_train, y_train)
        return trained_model

    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e