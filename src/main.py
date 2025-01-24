from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
from data_processer import load_and_prepare_data
from sklearn.metrics import mean_squared_error
import numpy as np


def train_model(regressor_cls=RandomForestRegressor, **regressor_params):
    X_train, y_train, X_test, y_test = load_and_prepare_data()
    model = regressor_cls(**regressor_params)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse_gold = np.sqrt(mean_squared_error(y_test.iloc[:, 0], predictions[:, 0]))
    rmse_total = np.sqrt(mean_squared_error(y_test.iloc[:, 1], predictions[:, 1]))
    return {"RMSE_Gold": rmse_gold, "RMSE_Total": rmse_total}

def predict_country_medals(model):
    import pandas as pd
    df = pd.read_csv("data/summerOly_medal_counts.csv")
    return {}

def main():
    model, X_test, y_test = train_model(RandomForestRegressor, n_estimators=100, random_state=42)
    results = evaluate_model(model, X_test, y_test)
    print(results)


if __name__ == "__main__":
    main()
