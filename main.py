import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from model_training import train_linear_regression, train_decision_tree, train_random_forest

def setup_logging():
    logging.basicConfig(filename='model_training.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    setup_logging()
    
    try:
        # Load the data
        df = pd.read_csv('final.csv')
        logging.info('Data loaded successfully.')
        
        # Separate input features and target variable
        x = df.drop('price', axis=1)
        y = df['price']

        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
        logging.info(f'Dataset split into train and test sets. Shapes: x_train: {x_train.shape}, x_test: {x_test.shape}')
        
        # Train models and evaluate
        train_linear_regression(x_train, x_test, y_train, y_test)
        train_decision_tree(x_train, x_test, y_train, y_test)
        train_random_forest(x_train, x_test, y_train, y_test)

    except Exception as e:
        logging.error(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
