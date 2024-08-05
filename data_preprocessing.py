import pandas as pd
import logging

# Import the logger setup
from logging_config import setup_logger

setup_logger()

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
        return None

def preprocess_data(df):
    """Preprocess the data."""
    try:
        # Add your preprocessing steps here
        df = df.dropna()  # Example step
        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return None
