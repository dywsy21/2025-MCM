from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def run_sensitivity_analysis(X, y):
    """
    Vary n_estimators across [50, 100, 150], train multiple models (like in main.py),
    select the best model for each medal type by MAE, compute an average MAE across
    medal types, and compare to the baseline (n_estimators=100) to produce two columns:
    (Inner_param_change_proportion, Result_change_proportion).
    """
    from sklearn.model_selection import KFold
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    import xgboost as xgb
    import pandas as pd
    import numpy as np

    # Define parameter values to explore
    # n_estimators_values = [50, 100, 150]
    n_estimators_values = [100 + 5 * x for x in range(-15, 16)]
    # We'll store mean MAE for each param setting (averaged over all medal types).
    # Then we compare each setting to the baseline (n_estimators=100).
    param_mae_map = {}
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    medal_types = ['Bronze', 'Silver', 'Gold']

    def train_and_get_avg_mae(n_value):
        """Train multiple models for each medal type, pick the best by MAE, return average MAE."""
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=n_value),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=n_value),
            'LinearRegression': LinearRegression(),  # no n_estimators parameter but included
            'XGBoost': xgb.XGBRegressor(n_estimators=n_value),
            # You can add more models here if desired
        }
        total_mae_for_medals = 0.0
        for medal_idx in range(len(medal_types)):
            best_score = float('inf')
            y_medal = y[:, medal_idx]
            # For each model, do K-Fold and pick best by MAE
            for name, base_model in models.items():
                fold_maes = []
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y_medal[train_idx], y_medal[val_idx]
                    cloned_model = clone(base_model)
                    cloned_model.fit(X_tr, y_tr)
                    preds = cloned_model.predict(X_val)
                    fold_maes.append(mean_absolute_error(y_val, preds))
                avg_mae = np.mean(fold_maes)
                if avg_mae < best_score:
                    best_score = avg_mae
            total_mae_for_medals += best_score
        # Return average across the three medal types
        return total_mae_for_medals / len(medal_types)

    # Collect average MAE for each n_estimators setting
    for n_value in n_estimators_values:
        param_mae_map[n_value] = train_and_get_avg_mae(n_value)

    # Define baseline n_estimators as 100
    baseline_param = 100
    baseline_mae = param_mae_map[baseline_param]

    # Build a DataFrame with just two columns:
    # Inner_param_change_proportion, Result_change_proportion
    rows = []
    for n_value in n_estimators_values:
        param_change = (n_value - baseline_param) / float(baseline_param)
        mae_change = (param_mae_map[n_value] - baseline_mae) / baseline_mae
        rows.append({
            'Inner_param_change_proportion': param_change,
            'Result_change_proportion': mae_change
        })

    pd.DataFrame(rows).to_csv('sensitivity_analysis_results.csv', index=False)
    print("Sensitivity analysis completed. Results saved to sensitivity_analysis_results.csv.")
