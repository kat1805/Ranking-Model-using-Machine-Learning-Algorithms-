import pandas as pd
import sys, json

#------------------------------------------------------------------------------
# Define relative target directories

paths_arg = sys.argv[1]

with open(paths_arg) as json_paths: 
    PATHS = json.load(json_paths)
    json_paths.close()

raw_internal_path = PATHS['raw_internal_data_path']
curated_data_path = PATHS['curated_data_path']
external_data_path = PATHS['external_data_path']

#------------------------------------------------------------------------------
# Retreive model metrics

BNPL_metrics = pd.read_csv(curated_data_path + "BNPL_metrics.csv")
BNPL_features = pd.read_csv(curated_data_path + "BNPL_features.csv")

customer_metrics = pd.read_csv(curated_data_path + "customer_metrics.csv")
customer_features = pd.read_csv(curated_data_path + "customer_features.csv")

revenue_metrics = pd.read_csv(curated_data_path + "revenue_metrics.csv")
revenue_features = pd.read_csv(curated_data_path + "revenue_features.csv")

transactions_metrics = pd.read_csv(
    curated_data_path + "transactions_metrics.csv")
transactions_features = pd.read_csv(
    curated_data_path + "transactions_features.csv")

metrics = {
    'Model name': ['Predicted BNPL earnings', 'Predicted no. of Customers', 
                        'Predicted Merchant Revenue', 
                        'Predicted no. of Transactions'],
    'Mean Absolute Error': [BNPL_metrics['MAE'].values[0], 
                        customer_metrics['MAE'].values[0],
                        revenue_metrics['MAE'].values[0],
                        transactions_metrics['MAE'].values[0]],
    'RMSE': [BNPL_metrics['RMSE'].values[0], customer_metrics['RMSE'].values[0],
                        revenue_metrics['RMSE'].values[0],
                        transactions_metrics['RMSE'].values[0]]
}

metrics_df = pd.DataFrame(metrics)


BNPL_features['name'][0]
feature = {
    'Model name': ['Predicted BNPL earnings', 'Predicted no. of Customers', 
                        'Predicted Merchant Revenue',
                        'Predicted no. of Transactions'],
    'First Feature': [BNPL_features['name'][0], customer_features['name'][0],
                revenue_features['name'][0], transactions_features['name'][0]],
    'Second Feature': [BNPL_features['name'][1],customer_features['name'][1],
                revenue_features['name'][1],transactions_features['name'][1]],
    'Third Feature': [BNPL_features['name'][2], customer_features['name'][2],
                revenue_features['name'][2], transactions_features['name'][2]]
}

feature_df = pd.DataFrame(feature)

