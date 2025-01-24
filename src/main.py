from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
from data_processer import load_and_prepare_data
from sklearn.metrics import mean_squared_error
import numpy as np

# Input data format:
# id(combined from athlete name and event name), (Gold, Silver, Bronze, Host_country) * 8
# 依次写出该运动员在前8届奥运会上在该项目的金牌数、银牌数、铜牌数、主办国
# 金银铜牌数只能为0, 1

# Use


if __name__ == "__main__":
    main()
