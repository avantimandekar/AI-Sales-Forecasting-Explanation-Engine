import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

feat = pd.read_csv(
    "clean_data/features.csv",
    parse_dates=["date"]
)

feature_cols = [
    "day_of_week",
    "is_weekend",
    "month",
    "days_since_start",
    "cancel_return_rate",
    "discount_rate",
    "top_category_share",
    "revenue_lag_1",
    "revenue_lag_7",
    "revenue_lag_14",
    "revenue_roll7_mean",
    "revenue_roll7_std",
    "revenue_roll30_mean"
]

split_date = feat["date"].max() - pd.Timedelta(days=56)

train = feat[
    feat["date"] <= split_date
]

test = feat[
    feat["date"] > split_date
]

X_train = train[feature_cols]
y_train = train["revenue"]

X_test = test[feature_cols]
y_test = test["revenue"]

baseline_pred = X_test["revenue_lag_7"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_mape = mean_absolute_percentage_error(
    y_test,
    baseline_pred
) * 100

print(
    "Baseline MAE:",
    round(baseline_mae, 2)
)

print(
    "Baseline MAPE:",
    round(baseline_mape, 1),
    "%"
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    pred
)

mape = mean_absolute_percentage_error(
    y_test,
    pred
) * 100

print(
    "Random Forest MAE:",
    round(mae, 2)
)

print(
    "Random Forest MAPE:",
    round(mape, 1),
    "%"
)

importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)

joblib.dump(
    model,
    "outputs/forecast_model.pkl"
)



