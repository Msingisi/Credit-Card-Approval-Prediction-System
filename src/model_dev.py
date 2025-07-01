import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass


class RandomForestModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            model = RandomForestClassifier(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Trained RandomForestClassifier.")
            return model
        except Exception as e:
            logging.error(f"Error training RandomForest: {e}")
            raise e


class GradientBoostingModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            model = GradientBoostingClassifier(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Trained GradientBoostingClassifier.")
            return model
        except Exception as e:
            logging.error(f"Error training GradientBoosting: {e}")
            raise e


class MLPModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            model = MLPClassifier(max_iter=1000, **kwargs)
            model.fit(X_train, y_train)
            logging.info("Trained MLPClassifier.")
            return model
        except Exception as e:
            logging.error(f"Error training MLPClassifier: {e}")
            raise e


class LogisticRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            model = LogisticRegression(max_iter=1000, **kwargs)
            model.fit(X_train, y_train)
            logging.info("Trained LogisticRegression.")
            return model
        except Exception as e:
            logging.error(f"Error training LogisticRegression: {e}")
            raise e


class XGBoostModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
            model.fit(X_train, y_train)
            logging.info("Trained XGBoost.")
            return model
        except Exception as e:
            logging.error(f"Error training XGBoost: {e}")
            raise e