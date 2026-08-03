from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess_reg import clean_data, get_feature_target, load_raw_data


def train_reg_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",
        subsample=0.6,
        reg_lambda=5.0,
        reg_alpha=1.0,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.08,
        colsample_bytree=1.0,
    )
    model.fit(X_train, y_train)

    preds_log = model.predict(X_test)

    # Log metrics
    r2 = r2_score(y_test, preds_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, preds_log))
    mae_log = mean_absolute_error(y_test, preds_log)

    # Dollar metrics ($M)
    y_test_usd = np.expm1(y_test)
    preds_usd = np.expm1(preds_log)
    abs_err = np.abs(y_test_usd - preds_usd)

    mae_m = abs_err.mean() / 1e6
    median_ae_m = np.median(abs_err) / 1e6

    print(f"R2 Score:        {r2:.4f}")
    print(f"RMSE (log):      {rmse_log:.4f}")
    print(f"MAE (log):       {mae_log:.4f}")
    print(f"MAE ($M):        ${mae_m:,.2f}M")
    print(f"Median AE ($M):  ${median_ae_m:,.2f}M")

    return model


if __name__ == "__main__":
    raw_data_path = "data/raw/Unicorn_Companies.csv"
    model_output_path = "models/xgb_unicorn_model.pkl"

    df = load_raw_data(raw_data_path)
    df = clean_data(df)
    X, y = get_feature_target(df)

    model = train_reg_model(X, y)

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModel saved successfully to '{model_output_path}'")