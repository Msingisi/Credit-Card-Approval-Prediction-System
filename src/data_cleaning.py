import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()

            # Drop irrelevant columns
            drop_cols = ['ID', 'Has a mobile phone', 'Children count', 'Job title', 'Account age']
            df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

            # Convert Age and Employment length from days to positive years
            if 'Age' in df.columns:
                df['Age'] = (-df['Age'] / 365.25)
            if 'Employment length' in df.columns:
                df['Employment length'] = (-df['Employment length'] / 365.25)
                df = df[df['Employment length'] >= 0]

            # Remove outliers using IQR for selected columns
            iqr_cols = ['Family member count', 'Income', 'Employment length']
            for col in iqr_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower) & (df[col] <= upper)]

            # Handle skewness using cube root transformation
            for col in ['Income', 'Age']:
                if col in df.columns:
                    df[col] = np.cbrt(df[col])

            # Ordinal Encoding for Education level
            if 'Education level' in df.columns:
                education_order = [['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']]
                encoder = OrdinalEncoder(categories=education_order)
                df[['Education level']] = encoder.fit_transform(df[['Education level']])

            # One-hot encode selected categorical columns
            ohe_cols = ['Gender', 'Marital status', 'Dwelling', 'Employment status', 
                        'Has a car', 'Has a property', 'Has a work phone', 
                        'Has a phone', 'Has an email']
            df = pd.get_dummies(df, columns=[col for col in ohe_cols if col in df.columns])

            # Min-Max Scaling for selected numerical columns
            scale_cols = ['Age', 'Income', 'Employment length']
            scaler = MinMaxScaler()
            for col in scale_cols:
                if col in df.columns:
                    df[[col]] = scaler.fit_transform(df[[col]])

            # Split features and target
            X = df.drop("Is high risk", axis=1)
            y = df["Is high risk"]

            # Class balance before SMOTE
            print("\nClass distribution BEFORE SMOTE:")
            print(Counter(y))

            # Apply SMOTE for imbalance correction
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Class balance after SMOTE
            print("\nClass distribution AFTER SMOTE:")
            print(Counter(y_resampled))

            # Recombine into a single DataFrame
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled['Is high risk'] = y_resampled

            logging.info("Data preprocessing and SMOTE complete.")
            return df_resampled

        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train  test
        """
        try:
            X = data.drop(["Is high risk"], axis=1)
            y = data["Is high risk"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise


class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e