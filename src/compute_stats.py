from preprocess_reg import TOP_CATEGORIES, clean_data, load_raw_data

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from preprocess_reg import get_feature_target

RAW_DATA_PATH = "data/raw/Unicorn_Companies.csv"

df = load_raw_data(RAW_DATA_PATH)
df = clean_data(df)

overall_mean = df["funding_usd_log"].mean()

print("CATEGORY_AVG_LOG_FUNDING = {")
zero_data_cats = []
for cat in TOP_CATEGORIES:
    cat_rows = df[df["mapped_category"] == cat]
    n = len(cat_rows)
    if n == 0:
        zero_data_cats.append(cat)
        avg = overall_mean
    else:
        avg = cat_rows["funding_usd_log"].mean()
    print(f'    "{cat}": {avg:.4f},')
print("}")
print(f"DEFAULT_AVG_LOG_FUNDING = {overall_mean:.4f}")

if zero_data_cats:
    print(
        f"\n# WARNING: these categories have 0 training rows and fell back "
        f"to the dataset-wide average -- valuations for them are "
        f"extrapolation: {zero_data_cats}"
    )

X, y = get_feature_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = xgb.XGBRegressor(
    random_state=42,
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
rmse_log = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print(f"\nRMSE_MARGIN = {rmse_log:.4f}")