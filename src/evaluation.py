import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.metrics import classification_report


class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating classification models.
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the score for the model.

        Args:
            y_true: True class labels
            y_pred: Predicted class labels or probabilities

        Returns:
            score (float)
        """
        pass


class Accuracy(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Accuracy")
            score = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy: {score}")
            return score
        except Exception as e:
            logging.error(f"Error in calculating Accuracy: {e}")
            raise e


class Precision(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Precision")
            score = precision_score(y_true, y_pred)
            logging.info(f"Precision: {score}")
            return score
        except Exception as e:
            logging.error(f"Error in calculating Precision: {e}")
            raise e


class Recall(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Recall")
            score = recall_score(y_true, y_pred)
            logging.info(f"Recall: {score}")
            return score
        except Exception as e:
            logging.error(f"Error in calculating Recall: {e}")
            raise e


class F1(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating F1 Score")
            score = f1_score(y_true, y_pred)
            logging.info(f"F1 Score: {score}")
            return score
        except Exception as e:
            logging.error(f"Error in calculating F1 Score: {e}")
            raise e


class ROC_AUC(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        """
        Note: y_pred_probs should be probabilities, not class labels.
        """
        try:
            logging.info("Calculating ROC AUC")
            score = roc_auc_score(y_true, y_pred_probs)
            logging.info(f"ROC AUC: {score}")
            return score
        except Exception as e:
            logging.error(f"Error in calculating ROC AUC: {e}")
            raise e
        
class ClassificationReport(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        try:
            logging.info("Generating Classification Report")
            report = classification_report(y_true, y_pred, target_names=["No Risk", "High Risk"])
            logging.info("\n" + report)
            return report
        except Exception as e:
            logging.error(f"Error generating classification report: {e}")
            raise e