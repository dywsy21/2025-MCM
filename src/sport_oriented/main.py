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
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone

def train_model(X_train, y_train):
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
        
        y_train_medal = y_train[:, medal_idx]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        best_score = float('inf')
        best_model_name = None
        best_model_instance = None
        
        for name, base_model in models.items():
            total_score = 0.0
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train_medal[train_idx], y_train_medal[val_idx]
                
                cloned_model = clone(base_model)
                cloned_model.fit(X_tr, y_tr)
                preds = cloned_model.predict(X_val)
                total_score += mean_absolute_error(y_val, preds)
            
            avg_score = total_score / kf.get_n_splits()
            if avg_score < best_score:
                best_score = avg_score
                best_model_name = name
                best_model_instance = clone(base_model)
        
        # Retrain the best model on the entire X_train
        best_model_instance.fit(X_train, y_train_medal)
        trained_models[medal_type] = {best_model_name: best_model_instance}
    
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
            
            # Calculate prediction intervals using residuals
            residuals = y_test_medal - predictions
            std_residuals = np.std(residuals)
            lower_bound = predictions - 1.96 * std_residuals
            upper_bound = predictions + 1.96 * std_residuals
            
            # Add validation checks
            if np.any(np.isnan(predictions)):
                print(f"Warning: NaN predictions found for {name} on {medal_type}")
                continue
                
            if np.all(predictions == 0):
                print(f"Warning: All zero predictions for {name} on {medal_type}")
                
            # Print some statistics
            print(f"\n{name} - {medal_type}:")
            print(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
            print(f"Mean prediction: {predictions.mean():.2f}")
            print(f"Actual range: {y_test_medal.min():.2f} to {y_test_medal.max():.2f}")
            print(f"Actual mean: {y_test_medal.mean():.2f}")
            
            mse = mean_squared_error(y_test_medal, predictions)
            mae = mean_absolute_error(y_test_medal, predictions)
            
            medal_results[name] = {
                'MSE': mse, 
                'MAE': mae,
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"95% Prediction Interval: ±{1.96 * std_residuals:.2f}")
        
        results[medal_type] = medal_results
    
    return results

def select_best_model(evaluation_results, metric='MAE'):
    """Select best model based on specified metric (MAE or MSE)"""
    best_models = {}
    for medal_type in ['Bronze', 'Silver', 'Gold']:
        best_score = float('inf')
        best_model_name = None
        
        for model_name, metrics in evaluation_results[medal_type].items():
            score = metrics[metric]
            if score < best_score:
                best_score = score
                best_model_name = model_name
        
        best_models[medal_type] = best_model_name
        print(f"\nBest model for {medal_type} ({metric}={best_score:.4f}): {best_model_name}")
    
    return best_models

def main():
    # Load data
    data = load_data('data/generated_training_data/sport_oriented/Training_data_tier.csv')
    
    # Get years to use for training (NUMBER_OF_MATCHES_TO_USE most recent Olympics before TARGET_YEAR - 4)
    years = sorted(data['Year'].unique())
    evaluation_year = TARGET_YEAR - 4
    training_years = [year for year in years if year < evaluation_year][-NUMBER_OF_MATCHES_TO_USE:]
    
    # Filter data to only include training years and evaluation year
    training_data = data[data['Year'].isin(training_years + [evaluation_year])]
    
    print(f"\nUsing Olympics from years {training_years} to predict {evaluation_year}")
    print(f"Then using that model to predict {TARGET_YEAR}")
    
    # Create features and labels using only the filtered data
    X_train, y_train = create_features(training_data, TARGET_YEAR, NUMBER_OF_MATCHES_TO_USE)
    
    # Get prediction data ready first
    prediction_years = training_years[-NUMBER_OF_MATCHES_TO_USE:] + [evaluation_year]
    prediction_data = data[data['Year'].isin(prediction_years)]
    X_pred = prepare_prediction_features(prediction_data, TARGET_YEAR, NUMBER_OF_MATCHES_TO_USE)
    
    # Verify feature consistency
    train_cols = set(X_train.columns) - {'NOC', 'Year', 'Host', 'Tier'}
    pred_cols = set(X_pred.columns) - {'NOC', 'Year', 'Host', 'Tier'}
    assert train_cols == pred_cols, "Feature mismatch between training and prediction data"
    
    # Verify no NaN values in features and labels
    assert not X_train.isnull().values.any(), "NaN values found in training features"
    assert not np.isnan(y_train).any(), "NaN values found in training labels"
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(['NOC', 'Year', 'Host', 'Tier'], axis=1))
    X_pred_scaled = scaler.transform(X_pred.drop(['NOC', 'Year', 'Host', 'Tier'], axis=1))
    
    # Split data into training and validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, 
        test_size=0.2, 
        random_state=42
    )
    
    # Train models on training split
    trained_models = train_model(X_train_split, y_train_split)
    
    # Evaluate models on validation split
    evaluation_results = evaluate_model(trained_models, X_val_split, y_val_split)
    
    # Select best models based on MAE
    best_models = select_best_model(evaluation_results, metric='MAE')
    
    # Make predictions using best models for each medal type
    predictions = {}
    prediction_intervals = {}
    for medal_type in ['Bronze', 'Silver', 'Gold']:
        best_model_name = best_models[medal_type]
        model = trained_models[medal_type][best_model_name]
        preds = model.predict(X_pred_scaled)
        
        # Calculate prediction intervals for final predictions
        train_preds = model.predict(X_train_scaled)
        residuals = y_train[:, ['Bronze', 'Silver', 'Gold'].index(medal_type)] - train_preds
        std_residuals = np.std(residuals)
        
        predictions[medal_type] = preds
        prediction_intervals[medal_type] = {
            'lower': preds - 1.96 * std_residuals,
            'upper': preds + 1.96 * std_residuals
        }
    
    # Create results dataframe
    results = pd.DataFrame({
        'NOC': X_pred['NOC'],
        'Year': X_pred['Year'],
        'Predicted_Bronze': predictions['Bronze'],
        'Predicted_Silver': predictions['Silver'],
        'Predicted_Gold': predictions['Gold'],
        'Bronze_Lower': prediction_intervals['Bronze']['lower'],
        'Bronze_Upper': prediction_intervals['Bronze']['upper'],
        'Silver_Lower': prediction_intervals['Silver']['lower'],
        'Silver_Upper': prediction_intervals['Silver']['upper'],
        'Gold_Lower': prediction_intervals['Gold']['lower'],
        'Gold_Upper': prediction_intervals['Gold']['upper']
    })
    
    results['Total_Medals'] = results[['Predicted_Bronze', 'Predicted_Silver', 'Predicted_Gold']].sum(axis=1)
    results = results.sort_values('Total_Medals', ascending=False)

    print("\nResults with prediction intervals:")
    print(results.to_string(index=False))
    
    # save to csv
    results.to_csv(f'sport_oriented_predictions_{TARGET_YEAR}.csv', index=False)

    # Adjust predictions to match total medal counts (without rounding)
    for medal_type, total in zip(['Bronze', 'Silver', 'Gold'], [BRONZE_TOTAL, SILVER_TOTAL, GOLD_TOTAL]):
        predicted_total = results[f'Predicted_{medal_type}'].sum()
        coefficient = total / predicted_total
        results[f'Predicted_{medal_type}'] *= coefficient
        results[f'{medal_type}_Lower'] *= coefficient
        results[f'{medal_type}_Upper'] *= coefficient
    
    # Update total medals
    results['Total_Medals'] = results[['Predicted_Bronze', 'Predicted_Silver', 'Predicted_Gold']].sum(axis=1)
    results = results.sort_values('Total_Medals', ascending=False)
    
    # Save predictions
    results.to_csv(f'sport_oriented_predictions_{TARGET_YEAR}_rounded.csv', index=False)
    print(f"\nPredictions for {TARGET_YEAR} saved to predictions/sport_oriented_predictions_{TARGET_YEAR}.csv")
    
    print("\nFinal results with prediction intervals:")
    print(results.to_string(index=False))

if __name__ == '__main__':
    main()
