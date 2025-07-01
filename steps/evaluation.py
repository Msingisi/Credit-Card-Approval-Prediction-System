import logging
from typing import Tuple
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, 
ConfusionMatrixDisplay, roc_curve, auc, classification_report
)
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

# Get experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
    Annotated[float, "f1"],
    Annotated[float, "roc_auc"],
]:
    """
    Evaluate the classification model using standard metrics and visualizations.
    Also logs metrics to MLflow.
    """
    try:
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Scores
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probs)
        report = classification_report(y_test, y_pred, target_names=["No Risk", "High Risk"])

        # Log scores
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"ROC AUC: {roc_auc}")
        logging.info(f"\nClassification Report:\n{report}")

        # Plot
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_probs)

        return accuracy, precision, recall, f1, roc_auc

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e