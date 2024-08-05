import pickle
import logging

# Import the logger setup
from logging_config import setup_logger

setup_logger()

def save_model(model, filename):
    """Save a trained model to a file."""
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved to {filename}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def load_model(filename):
    """Load a model from a file."""
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded from {filename}.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
