import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(target_year):
    data = pd.read_csv('data/generated_training_data/training_data.csv')

    # Prepare the data
    # Assuming the data has columns: 'id', 'Gold_{year}', 'Silver_{year}', 'Bronze_{year}', ...
    years = [target_year - i*4 for i in range(1, 9)]
    feature_columns = [f'Gold_{year}' for year in years] + [f'Silver_{year}' for year in years] + [f'Bronze_{year}' for year in years] + [f'Host_country_{year}' for year in years]
    
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

