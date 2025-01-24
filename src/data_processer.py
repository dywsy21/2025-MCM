import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
import hashlib

def hash_noc(noc):
    return int(hashlib.sha256(noc.encode('utf-8')).hexdigest(), 16) % 10**8

def load_and_prepare_data(target_year, linear_transformation=False):
    data = pd.read_csv('data/generated_training_data/ult.csv')
    
    # Invalidate a player if target_year > First_Year_In_Match + RETIREMENT_LIMIT
    data = data[data['First_Year_In_Match'] + RETIREMENT_LIMIT >= target_year]

    # Convert NOC strings to numbers using hashing
    data['NOC'] = data['NOC'].apply(hash_noc)
    data['id'] = data['id'].apply(hash_noc)
    
    years = [target_year - i * 4 for i in range(1, NUMBER_OF_MATCHES_TO_USE + 1)]

    # Apply linear transformation
    if linear_transformation:
        for year in years:
            data.loc[:, f'Gold_{year}'] = data[f'Gold_{year}'].apply(lambda x: x * 100 + 10)
            data.loc[:, f'Silver_{year}'] = data[f'Silver_{year}'].apply(lambda x: x * 100 + 10)
            data.loc[:, f'Bronze_{year}'] = data[f'Bronze_{year}'].apply(lambda x: x * 100 + 10)

    # Prepare the data
    # Assuming the data has columns: 'id', 'Country', 'Gold_{year}', 'Silver_{year}', 'Bronze_{year}', ...
    feature_columns = ['NOC'] + [f'Gold_{year}' for year in years] + [f'Silver_{year}' for year in years] + [f'Bronze_{year}' for year in years] + [f'Host_NOC_{year}' for year in years]
    
    features = data[['id'] + feature_columns]
    
    # Targets for gold, silver, and bronze medals
    target_gold = data[f'Gold_{target_year}']
    target_silver = data[f'Silver_{target_year}']
    target_bronze = data[f'Bronze_{target_year}']
    
    # Split the data into training and testing sets for each target
    # Training set: data up to 2020
    # Testing set: data for 2024
    train_data = data[data['Year'] < target_year]
    test_data = data[data['Year'] == target_year]
    
    X_train_gold = train_data[feature_columns]
    y_train_gold = train_data[f'Gold_{target_year}']
    X_test_gold = test_data[feature_columns]
    y_test_gold = test_data[f'Gold_{target_year}']
    
    X_train_silver = train_data[feature_columns]
    y_train_silver = train_data[f'Silver_{target_year}']
    X_test_silver = test_data[feature_columns]
    y_test_silver = test_data[f'Silver_{target_year}']
    
    X_train_bronze = train_data[feature_columns]
    y_train_bronze = train_data[f'Bronze_{target_year}']
    X_test_bronze = test_data[feature_columns]
    y_test_bronze = test_data[f'Bronze_{target_year}']
    
    return {
        'gold': {
            'train': (X_train_gold, y_train_gold),
            'test': (X_test_gold, y_test_gold)
        },
        'silver': {
            'train': (X_train_silver, y_train_silver),
            'test': (X_test_silver, y_test_silver)
        },
        'bronze': {
            'train': (X_train_bronze, y_train_bronze),
            'test': (X_test_bronze, y_test_bronze)
        }
    }

def load_and_prepare_data_normalization(target_year):
    data = pd.read_csv('data/generated_training_data/ult_value.csv')
    
    # Invalidate a player if target_year > First_Year_In_Match + RETIREMENT_LIMIT
    data = data[data['First_Year_In_Match'] + RETIREMENT_LIMIT >= target_year]
    

    # Convert NOC strings to numbers using hashing
    data['NOC'] = data['NOC'].apply(hash_noc)
    data['id'] = data['id'].apply(hash_noc)
    
    years = [target_year - i * 4 for i in range(1, NUMBER_OF_MATCHES_TO_USE + 1)]

    # Prepare the data
    # Assuming the data has columns: 'id', 'Country', 'Gold_{year}', 'Silver_{year}', 'Bronze_{year}', ...
    feature_columns = [f'Host_NOC_{year}' for year in years] + ['NOC'] + ['id']
    
    features = data[feature_columns]
    
    # Targets for gold, silver, and bronze medals
    target_gold = data[f'Gold_{target_year}']
    target_silver = data[f'Silver_{target_year}']
    target_bronze = data[f'Bronze_{target_year}']
    
    # Split the data into training and testing sets for each target
    X_train_gold, X_test_gold, y_train_gold, y_test_gold = train_test_split(features, target_gold, test_size=0.2, random_state=42)
    X_train_silver, X_test_silver, y_train_silver, y_test_silver = train_test_split(features, target_silver, test_size=0.2, random_state=42)
    X_train_bronze, X_test_bronze, y_train_bronze, y_test_bronze = train_test_split(features, target_bronze, test_size=0.2, random_state=42)
    
    return {
        'gold': {
            'train': (X_train_gold, y_train_gold),
            'test': (X_test_gold, y_test_gold)
        },
        'silver': {
            'train': (X_train_silver, y_train_silver),
            'test': (X_test_silver, y_test_silver)
        },
        'bronze': {
            'train': (X_train_bronze, y_train_bronze),
            'test': (X_test_bronze, y_test_bronze)
        }
    }


