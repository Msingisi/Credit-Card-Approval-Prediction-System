import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy

@step
def clean_df(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Cleans and splits the data into train and test sets.
    
    Args:
        df: Raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        # Preprocessing
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        # Splitting
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e
