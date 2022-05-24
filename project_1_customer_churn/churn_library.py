"""
Author - Sebastien
Date - 2022/05/24

Reads customer churn data, performs exploratory data analysis and
trains models for predicting likelihood of customer churn
"""

# Packages
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
plt.style.use('ggplot')

# Global Variables
SEED = 42
idx = pd.IndexSlice
# os.environ['QT_QPA_PLATFORM']='offscreen'

# Logging
logger = logging.getLogger('churn_library')
logger.setLevel(logging.INFO)
file_logger = logging.FileHandler(filename='logs/churn_library.log')
file_logger.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
file_logger.setFormatter(file_format)
logger.addHandler(file_logger)

# Functions


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        # Check if file path is of CSV format
        assert pth.rsplit('.', 1)[-1] == 'csv'

        # Read dataset
        data = pd.read_csv(pth, index_col=0)

        # Run some basic preprocessing
        data['Churn'] = data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        logger.info('Imported Dataset')

    except FileNotFoundError:
        logger.error(f'Path provided {pth} either does not exist.')
        raise FileNotFoundError

    except AssertionError:
        logger.error(f'Path provided {pth} is not of CSV format.')
        raise AssertionError

    return data


def perform_eda(df: pd.DataFrame, output_path: str) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            output_path: folder to output images

    output:
            None
    '''
    # Plot Customer Churn Distribution
    _, ax = plt.subplots(figsize=(8, 5))
    df['Churn'].hist(ax=ax, color='blue')
    ax.set_title('Distribution of customer churn', fontsize=12)
    ax.set_xlabel('0 = Existing Customer; 1 = Customer Churn', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    plt.savefig(f'{output_path}/churn_distribution.png', bbox_inches='tight')
    plt.close()

    # Plot Customer Age Distribution
    _, ax = plt.subplots(figsize=(8, 5))
    df['Customer_Age'].hist(ax=ax, color='blue',
                            edgecolor='white')
    ax.set_title('Distribution of customer churn', fontsize=12)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    plt.savefig(f'{output_path}/age_distribution.png', bbox_inches='tight')
    plt.close()

    # Plot Marital Status Distribution
    _, ax = plt.subplots(figsize=(8, 5))
    df['Marital_Status'].value_counts(True) \
        .plot(kind='bar', ax=ax, color='blue')
    ax.set_title('Distribution of marital status', fontsize=12)
    ax.set_xlabel('Marital Status', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.savefig(f'{output_path}/marital_status_distribution.png',
                bbox_inches='tight')
    plt.close()

    # Plot Total Transaction Count
    _, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Total_Trans_Ct'], ax=ax, color='blue',
                 kde=True)
    ax.set_title('Distribution of total transaction counts', fontsize=12)
    ax.set_xlabel('Total Transaction Counts', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    plt.savefig(f'{output_path}/total_transaction_distribution.png',
                bbox_inches='tight')
    plt.close()

    # Plot Data Set Heatmap
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=False, ax=ax, linewidths=2,
                cmap='RdYlGn_r', vmin=-1.0, vmax=1.0)
    ax.set_title('Correlation map of customer churn dataset', fontsize=12)
    plt.savefig(f'{output_path}/correlation_map.png',
                bbox_inches='tight')
    plt.close()

    logger.info('Performed Exploratory Data Analysis')


def encoder_helper(df: pd.DataFrame,
                   category_lst: list,
                   response: str
                   ) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    data = df.copy()
    for category_feature in category_lst:

        # Dictionary containing average churn per category
        category_feature_dict = data.groupby(category_feature)[response].mean()
        category_feature_dict = category_feature_dict.to_dict()

        # Map categories with their average churn percentage
        new_feature = f'{category_feature}_Churn'
        data[new_feature] = data[category_feature].map(category_feature_dict)

    return data


def perform_feature_engineering(
        df: pd.DataFrame,
        response: str) -> pd.DataFrame:
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    data = df.copy()

    # Encoding Categorical Variables
    category_vars = ['Gender', 'Education_Level', 'Marital_Status',
                     'Income_Category', 'Card_Category']
    data_category_encoded = encoder_helper(data, category_vars,
                                           response='Churn')

    # Keep Relevant Variables
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = data_category_encoded[keep_cols].copy()
    y = data_category_encoded[response].copy()

    # Training/Testing Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=SEED)
    result = X_train, X_test, y_train, y_test

    logger.info('Performed Feature Engineering')

    return result


def classification_report_image(y_train: pd.DataFrame,
                                y_test: pd.DataFrame,
                                y_train_preds_lr: pd.DataFrame,
                                y_train_preds_rf: pd.DataFrame,
                                y_test_preds_lr: pd.DataFrame,
                                y_test_preds_rf: pd.DataFrame,
                                output_folder: str):
    '''
    produces classification report for training and testing results and stores report
    as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            output_folder: folder to output images

    output:
             None
    '''
    # Transform Probability Predictions into Predictions
    ytrain_preds_lr = (y_train_preds_lr >= 0.5).astype(int)
    ytrain_preds_rf = (y_train_preds_rf >= 0.5).astype(int)
    ytest_preds_lr = (y_test_preds_lr >= 0.5).astype(int)
    ytest_preds_rf = (y_test_preds_rf >= 0.5).astype(int)

    # Generate Random Forest Reports
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, ytrain_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')

    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, ytest_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        f'{output_folder}/random_forest_classification_report.png',
        bbox_inches='tight')
    plt.close()

    # Generate Logistic Regression Reports as Images
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, ytrain_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')

    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, ytest_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        f'{output_folder}/logistic_reg_classification_report.png',
        bbox_inches='tight')
    plt.close()


def roc_curve_plot(y_test: pd.DataFrame,
                   y_test_preds_lr: pd.DataFrame,
                   y_test_preds_rf: pd.DataFrame,
                   output_pth: str) -> None:
    '''
    creates roc curve plots for testing set and stores them as png
    images
    input:
         y_test:  test response values
         y_test_preds_lr: test predictions from logistic regression
         y_test_preds_rf: test predictions from random forest
         output_pth: path to store images
    output:
         None
    '''
    # Get ROC Curves for both models
    false_positive_lrc, true_positive_lrc, _ = roc_curve(
        y_test, y_test_preds_lr)
    false_positive_rfc, true_positive_rfc, _ = roc_curve(
        y_test, y_test_preds_rf)
    roc_auc_lrc = roc_auc_score(y_test, y_test_preds_lr)
    roc_auc_rfc = roc_auc_score(y_test, y_test_preds_rf)

    # Get Plots
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(false_positive_lrc, true_positive_lrc, color='blue',
            label=f'Logistic Regression (AUC: {roc_auc_lrc:.2f})')
    ax.plot(false_positive_rfc, true_positive_rfc, color='green',
            label=f'Random Forest (AUC: {roc_auc_rfc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', label='Random (AUC: 0.5)',
            ls='--')
    ax.legend(loc='best')
    ax.set_xlabel('False Positive Ratio', fontsize=12)
    ax.set_ylabel('True Positive Ratio', fontsize=12)
    ax.set_title('Testing Set ROC AUC Curves', fontsize=12)
    plt.savefig(f'{output_pth}/roc_curves.png', bbox_inches='tight')
    plt.close()


def feature_importance_plot(model_path: str,
                            X_data: pd.DataFrame,
                            output_pth: str):
    '''
    creates and stores the feature importances in pth
    input:
            model_path: path to serialized model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Load model
    rf_model = joblib.load(model_path)

    # Compute feature importances
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Plot Figure
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance", fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(f'{output_pth}/feature_importances.png',
                bbox_inches='tight')
    plt.close()


def train_models(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.DataFrame,
                 output_folder: str) -> list:
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              output_folder: folder to output models
    output:
              y_train_preds_lr: training predictions from logistic regression
              y_train_preds_rf: training predictions from random forest
              y_test_preds_lr: test predictions from logistic regression
              y_test_preds_rf: test predictions from random forest
    '''
    # Create Models
    rfc = RandomForestClassifier(random_state=SEED)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Train Random Forest (with Grid Search) and Get Predictions
    param_grid = {'n_estimators': [200, 500],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [4, 5, 100],
                  'criterion': ['gini', 'entropy']
                  }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict_proba(X_train)[:, 1]
    y_test_preds_rf = cv_rfc.best_estimator_.predict_proba(X_test)[:, 1]

    # Train Logistic Regression and Get Predictions
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict_proba(X_train)[:, 1]
    y_test_preds_lr = lrc.predict_proba(X_test)[:, 1]

    # Write Trained Models as Pickle Files
    best_rfc = cv_rfc.best_estimator_
    joblib.dump(best_rfc, f'{output_folder}/rfc_model.pkl')
    joblib.dump(lrc, f'{output_folder}/logistic_model.pkl')

    # Store results
    predictions_lst = y_train_preds_lr, y_train_preds_rf, \
        y_test_preds_lr, y_test_preds_rf

    logger.info('Trained Models')

    return predictions_lst


if __name__ == '__main__':

    command_line_arguments = sys.argv

    if len(command_line_arguments) == 2:

        # 1. Load Dataset
        data_path = command_line_arguments[1]
        data = import_data(data_path)

        # 2. Perform Exploratory Data Analysis
        perform_eda(data, 'images/eda')

        # 3. Data Preprocessing
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data, response='Churn'
        )

        # 4. Train Models
        model_predictions_lst = train_models(X_train, X_test, y_train,
                                             'models')
        y_train_preds_lr, y_train_preds_rf, \
            y_test_preds_lr, y_test_preds_rf = model_predictions_lst

        # 5. Metrics & Feature Importance
        classification_report_image(y_train, y_test, y_train_preds_lr,
                                    y_train_preds_rf, y_test_preds_lr,
                                    y_test_preds_rf, 'images/results')
        roc_curve_plot(y_test, y_test_preds_lr, y_test_preds_rf,
                       'images/results')

        feature_importance_plot('models/rfc_model.pkl', X_test,
                                'images/results')

    else:
        logger.error(
            'Wrong number of arguments provided. Only requires data path.')
