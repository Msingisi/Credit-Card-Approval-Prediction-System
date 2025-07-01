from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy for Data Types Inspection
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


# Concrete Strategy for Summary Statistics Inspection
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.
        Converts 'Age', 'Employment length' from days to positive years.
        Converts 'Account age' to positive months.
        Excludes negative employment lengths from summary stats.
        """
        df = df.copy()

        # Fix Age (from negative days to positive years)
        if 'Age' in df.columns:
            df['Age'] = (-df['Age'] / 365.25)

        # Fix Employment length (days to years)
        if 'Employment length' in df.columns:
            df['Employment length'] = (-df['Employment length'] / 365.25)
            # Remove invalid values (e.g., before employment started)
            df = df[df['Employment length'] >= 0]

        # Fix Account age (ensure it's positive)
        if 'Account age' in df.columns:
            df['Account age'] = df['Account age'].abs()

        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())

        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# Context Class that uses a DataInspectionStrategy
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)