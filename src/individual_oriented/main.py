from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sympy import use
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from data_processer import *
import pandas as pd
from config import *

# Input data format:
# id, (Gold, Silver, Bronze, Host_country) * 8
# id is combined from athlete name and event name
# 依次写出该运动员在前8届奥运会上在该项目的金牌数、银牌数、铜牌数、主办国
# 金银铜牌数只能为0, 1

# Training: use 1992~2020 data to predict 2024 medals
# Testing: use 2000~2024 data to predict 2028 medals

# Define and implement several auxiliary functions below, and then implement the main function.

def train_model(X_train, y_train):
    models = {
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'LinearRegression': LinearRegression(),
        'XGBoost': xgb.XGBRegressor(),
        'LightGBM': lgb.LGBMRegressor()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_model(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        results[name] = {'MSE': mse, 'MAE': mae, 'predictions': predictions}
    return results

def main_1(use_normalization=False):
    # Directly * 100 + 10 to increase k and avoid zero
    target_year = TARGET_YEAR  # Change this to the year you want to predict
    
    # Load and prepare data
    data = None
    
    if not use_normalization:
        data = load_and_prepare_data(target_year, linear_transformation=True)
    else:
        data = load_and_prepare_data_normalization(target_year)
    
    country_medals = {'gold': {}, 'silver': {}, 'bronze': {}}
    
    for medal_type in ['gold', 'silver', 'bronze']:
        print(f"Training and evaluating models for {medal_type} medals")
        
        X_train, y_train = data[medal_type]['train']
        X_test, y_test = data[medal_type]['test']

        # Train models
        models = train_model(X_train, y_train)

        # Evaluate models
        results = evaluate_model(models, X_test, y_test)
        best_model_name = min(results, key=lambda k: (results[k]['MSE'], results[k]['MAE']))
        best_model = models[best_model_name]
        print(f"Best model for {medal_type} medals: {best_model_name}")
        
        for name, metrics in results.items():
            print(f"{name} {medal_type} MSE: {metrics['MSE']}, MAE: {metrics['MAE']}")
        
        # Predict medals for each country
        predictions = results[best_model_name]['predictions']
        countries = X_test['NOC']
        
        if not use_normalization:
            for country, prediction in zip(countries, predictions):
                if country not in country_medals[medal_type]:
                    country_medals[medal_type][country] = 0
                country_medals[medal_type][country] += prediction
        else:
            for country, prediction in zip(countries, predictions):
                if 0 <= prediction <= 0.25:
                    pass
                elif 0.25 <= prediction <= 0.5:
                    if not country in country_medals['bronze'].keys():
                        country_medals['bronze'][country] = 0
                    country_medals['bronze'][country] += 1
                elif 0.5 <= prediction <= 0.75:
                    if not country in country_medals['silver'].keys():
                        country_medals['silver'][country] = 0
                    country_medals['silver'][country] += 1
                elif 0.75 <= prediction <= 1:
                    if not country in country_medals['gold'].keys():
                        country_medals['gold'][country] = 0
                    country_medals['gold'][country] += 1
                
    # Draw the prediction into a table
    table_data = []
    for medal_type, medals in country_medals.items():
        for country, count in medals.items():
            table_data.append({'Country': country, 'Medal_Type': medal_type, 'Count': count})
    
    df = pd.DataFrame(table_data)
    df.to_csv(f'projected_medals_of_{target_year}.csv', index=False)

if __name__ == "__main__":
    main_1(use_normalization=False)
