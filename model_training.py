import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import tree

def train_linear_regression(x_train, x_test, y_train, y_test):
    try:
        lrmodel = LinearRegression().fit(x_train, y_train)
        logging.info('Linear Regression model trained successfully.')

        # Evaluate Linear Regression model
        train_pred = lrmodel.predict(x_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        logging.info(f'Linear Regression Train MAE: {train_mae}')

        ypred = lrmodel.predict(x_test)
        test_mae = mean_absolute_error(y_test, ypred)
        logging.info(f'Linear Regression Test MAE: {test_mae}')

    except Exception as e:
        logging.error(f'Error training Linear Regression model: {e}')

def train_decision_tree(x_train, x_test, y_train, y_test):
    try:
        dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
        dtmodel = dt.fit(x_train, y_train)
        logging.info('Decision Tree model trained successfully.')

        # Evaluate Decision Tree model
        ytest_pred = dtmodel.predict(x_test)
        test_mae = mean_absolute_error(y_test, ytest_pred)
        logging.info(f'Decision Tree Test MAE: {test_mae}')

        ytrain_pred = dtmodel.predict(x_train)
        train_mae = mean_absolute_error(y_train, ytrain_pred)
        logging.info(f'Decision Tree Train MAE: {train_mae}')

        # Plot and save the tree
        plt.figure(figsize=(20,10))
        tree.plot_tree(dtmodel, feature_names=x_train.columns)
        plt.savefig('tree.png', dpi=300)
        logging.info('Decision Tree plot saved.')

    except Exception as e:
        logging.error(f'Error training Decision Tree model: {e}')

def train_random_forest(x_train, x_test, y_train, y_test):
    try:
        rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
        rfmodel = rf.fit(x_train, y_train)
        logging.info('Random Forest model trained successfully.')

        # Evaluate Random Forest model
        ytrain_pred = rfmodel.predict(x_train)
        ytest_pred = rfmodel.predict(x_test)
        test_mae = mean_absolute_error(y_test, ytest_pred)
        logging.info(f'Random Forest Test MAE: {test_mae}')

        # Save the trained Random Forest model
        with open('RE_Model.pkl', 'wb') as file:
            pickle.dump(rfmodel, file)
        logging.info('Random Forest model saved.')

        # Load the trained model
        with open('RE_Model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        # Example prediction
        example_input = np.array([2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]).reshape(1, -1)
        prediction = loaded_model.predict(example_input)
        logging.info(f'Prediction for example input: {prediction}')

    except Exception as e:
        logging.error(f'Error training Random Forest model: {e}')
