"""
Author - Sebastien
Date - 2022/05/24

Runs a few assert statements to test functions from customer churn library
"""

# Packages
import os
import logging
import pandas.api.types as pandas_types
from churn_library import import_data, perform_eda, \
    encoder_helper, perform_feature_engineering, train_models

# Setting up logger
logger = logging.getLogger('churn_library_test')
logger.setLevel(logging.INFO)

file_logger = logging.FileHandler(filename='logs/churn_test_library.log')
file_logger.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
file_logger.setFormatter(file_format)
logger.addHandler(file_logger)


# Functions
def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, import_data):
    '''
    test perform eda function
    '''
    try:
        df = import_data("data/bank_data.csv")
        _ = perform_eda(df, 'images/eda')
        nb_image_files = len(os.listdir('images/eda'))
        assert nb_image_files == 5
        logger.info('Testing perform_eda: SUCCESS')

    except AssertionError as err:
        logger.error(
            'Testing perform_eda: There should be 5 exploratory graphs in folder "images/eda"')
        raise err


def test_encoder_helper(encoder_helper, import_data):
    '''
    test encoder helper
    '''
    try:
        df = import_data('data/bank_data.csv')
        category_vars = ['Gender', 'Education_Level', 'Marital_Status',
                         'Income_Category', 'Card_Category']
        df_encoded = encoder_helper(df, category_vars, response='Churn')
        category_vars_new = [f'{col}_Churn' for col in category_vars]

        # Check if every categorical variable is now encoded
        # as a float
        assert all(pandas_types.is_numeric_dtype(df_encoded[col])
                   for col in category_vars_new)
        logger.info('Testing encoder_helper: SUCCESS')

    except AssertionError as err:
        logger.error(
            'Testing encoder_helper: Encoding failed. Not all columns are encoded as floats')
        logger.error(df_encoded[category_vars_new].dtypes)
        raise err


def test_perform_feature_engineering(perform_feature_engineering, import_data):
    '''
    test perform_feature_engineering
    '''
    try:
        df = import_data('data/bank_data.csv')
        X_train, _, _, _ = perform_feature_engineering(df, 'Churn')

        assert len(X_train.columns) == 19
        logger.info('Testing perform_feature_engineering: SUCCESS')

    except AssertionError as err:
        logger.error(
            'Testing perform_feature_engineering: Failed. There should be 19 features in dataset')
        raise err


def test_train_models(train_models, perform_feature_engineering, import_data):
    '''
    test train_models
    '''
    try:
        df = import_data('data/bank_data.csv')
        X_train, X_test, y_train, _ = perform_feature_engineering(df, 'Churn')
        _ = train_models(X_train, X_test, y_train, 'models')

        # Check if two trained models (Logistic Regression and
        # Random Forests) were serialized
        nb_models_trained = len(os.listdir('models/'))
        assert nb_models_trained == 2
        logger.info('Testing train_models: SUCCESS')

    except AssertionError as err:
        logger.error(
            'Testing train_models: Failed. There should be 2 serialized models')
        raise err


if __name__ == "__main__":

    test_import(import_data)
    test_eda(perform_eda, import_data)
    test_encoder_helper(encoder_helper, import_data)
    test_perform_feature_engineering(perform_feature_engineering, import_data)
    test_train_models(train_models, perform_feature_engineering, import_data)
