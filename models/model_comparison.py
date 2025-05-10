import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    BayesianRidge
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(df, years, target, model_ctor):
    """
    Perform expanding‐window CV on df for the given model constructor.
    Returns (avg_rmse, avg_r2, overall_rmse, overall_r2).
    """
    holdout_years = years[-min(2, len(years) - 1):]
    fold_rmses, fold_r2s = [], []
    all_y_true, all_y_pred = [], []

    for yr in holdout_years:
        train = df[df['year'] < yr]
        test  = df[df['year'] == yr]

        X_train = train.drop(columns=[target, 'country_text_id', 'year'])
        y_train = train[target]
        X_test  = test.drop(columns=[target, 'country_text_id', 'year'])
        y_test  = test[target]

        pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            model_ctor()
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_y_true.append(y_test.values)
        all_y_pred.append(y_pred)

    avg_rmse = np.mean(fold_rmses)
    avg_r2   = np.mean(fold_r2s)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    overall_r2   = r2_score(all_y_true, all_y_pred)

    return avg_rmse, avg_r2, overall_rmse, overall_r2

def main():
    # Load & clean
    df = pd.read_parquet('./data/subset_numeric.parquet')
    target = 'INDEX_v2x_corr'
    idx_to_drop = [c for c in df.columns if c.startswith('INDEX_') and c != target]
    df = df.drop(columns=idx_to_drop)

    years = sorted(df['year'].unique())

    # Models to compare
    models = {
        'OLS'             : LinearRegression,
        'Ridge'           : Ridge,
        'Lasso'           : Lasso,
        'Bayesian Ridge'  : BayesianRidge,
        'Random Forest'   : RandomForestRegressor,
        'Extra-Trees'     : ExtraTreesRegressor,
        'XGBoost'         : XGBRegressor
    }

    records = []
    for name, ctor in models.items():
        avg_rmse, avg_r2, overall_rmse, overall_r2 = evaluate_model(
            df, years, target, ctor
        )
        records.append({
            'Model':         name,
            'Avg RMSE':      avg_rmse,
            'Avg R²':        avg_r2,
            'Overall RMSE':  overall_rmse,
            'Overall R²':    overall_r2
        })

    perf_df = pd.DataFrame(records)
    print("\nModel comparison (2‐year expanding‐window CV):\n")
    print(perf_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()
