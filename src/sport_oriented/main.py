from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from config import *
from utils import *

def train_model(X_train, y_train):
    # Train separate models for Bronze, Silver, and Gold
    medal_types = ['Bronze', 'Silver', 'Gold']
    trained_models = {}
    
    for medal_idx, medal_type in enumerate(medal_types):
        models = {
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'LinearRegression': LinearRegression(),
            'XGBoost': xgb.XGBRegressor(),
            # 'LightGBM': lgb.LGBMRegressor()
        }
        
        # Get the column for current medal type
        y_train_medal = y_train[:, medal_idx]
        
        medal_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train_medal)
            medal_models[name] = model
            
        trained_models[medal_type] = medal_models
    
    return trained_models

def evaluate_model(models, X_test, y_test):
    results = {}
    medal_types = ['Bronze', 'Silver', 'Gold']
    
    for medal_type in medal_types:
        medal_results = {}
        medal_idx = medal_types.index(medal_type)
        y_test_medal = y_test[:, medal_idx]
        
        for name, model in models[medal_type].items():
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test_medal, predictions)
            mae = mean_absolute_error(y_test_medal, predictions)
            medal_results[name] = {'MSE': mse, 'MAE': mae, 'predictions': predictions}
        
        results[medal_type] = medal_results
    
    return results

def main():
    # Load data
    data = load_data('data/generated_training_data/sport_oriented/Training_data.csv')
    
    # Verify no NaN values in data
    assert not data.isnull().values.any(), "NaN values found in data"
    
    # Create features and labels for training
    X_train, y_train = create_features(data, TARGET_YEAR, NUMBER_OF_MATCHES_TO_USE)
    
    # Verify no NaN values in features and labels
    assert not X_train.isnull().values.any(), "NaN values found in training features"
    assert not np.isnan(y_train).any(), "NaN values found in training labels"
    
    # Split features for prediction
    X_pred = prepare_prediction_features(data, TARGET_YEAR, NUMBER_OF_MATCHES_TO_USE)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(['NOC', 'Year'], axis=1))
    X_pred_scaled = scaler.transform(X_pred.drop(['NOC', 'Year'], axis=1))
    
    # Train models
    trained_models = train_model(X_train_scaled, y_train)
    
    # Make predictions for target year
    predictions = {}
    for medal_type in ['Bronze', 'Silver', 'Gold']:
        model = trained_models[medal_type]['GradientBoosting']  # Using GradientBoosting as default
        preds = model.predict(X_pred_scaled)
        predictions[medal_type] = np.maximum(np.round(preds), 0).astype(int)  # Ensure non-negative integers
    
    # Create results dataframe
    results = pd.DataFrame({
        'NOC': X_pred['NOC'],
        'Year': X_pred['Year'],
        'Predicted_Bronze': predictions['Bronze'],
        'Predicted_Silver': predictions['Silver'],
        'Predicted_Gold': predictions['Gold']
    })
    
    # Sort by total predicted medals
    results['Total_Medals'] = results[['Predicted_Bronze', 'Predicted_Silver', 'Predicted_Gold']].sum(axis=1)
    results = results.sort_values('Total_Medals', ascending=False)
    
    # Save predictions
    results.to_csv(f'sport_oriented_predictions_{TARGET_YEAR}.csv', index=False)
    print(f"\nPredictions for {TARGET_YEAR} saved to predictions/sport_oriented_predictions_{TARGET_YEAR}.csv")
    
    # Display top 10 countries
    print("\nTop 10 predicted medal-winning countries:")
    print(results.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
