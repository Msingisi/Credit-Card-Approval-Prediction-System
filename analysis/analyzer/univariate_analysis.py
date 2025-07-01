from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass


# Concrete Strategy for Numerical Features
# -----------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """
        Plots a boxplot and histogram with KDE for numerical features.
        Applies transformations for 'Age', 'Employment length', and 'Account age'.
        """
        df = df.copy()
        feature_label = feature

        # Convert Age from days to positive years
        if feature == 'Age':
            df[feature] = (-df[feature] / 365.25)
            feature_label = "Age (years)"

        # Convert Employment length from days to positive years
        elif feature == 'Employment length':
            df[feature] = (-df[feature] / 365.25)
            df = df[df[feature] >= 0]  # Exclude invalid entries
            feature_label = "Employment Length (years)"

        # Convert Account age to positive months
        elif feature == 'Account age':
            df[feature] = df[feature].abs()

        fig, axs = plt.subplots(nrows=2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]})

        # Boxplot
        sns.boxplot(x=df[feature], ax=axs[0], color='skyblue')
        axs[0].set_title(f'Boxplot of {feature_label}')
        axs[0].set_xlabel('')

        # Histogram with KDE
        sns.histplot(df[feature], kde=True, bins=50, ax=axs[1], color='steelblue')
        axs[1].set_title(f'Distribution of {feature_label}')
        axs[1].set_xlabel(feature_label)
        axs[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()


# Concrete Strategy for Categorical Features
# -------------------------------------------
# This strategy analyzes categorical features by plotting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue=feature, data=df, palette="muted", legend=False)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# Context Class that uses a UnivariateAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature)
