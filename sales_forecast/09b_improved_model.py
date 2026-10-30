import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

feat = pd.read_csv("clean_data/features.csv", parse_dates=["date"])

feature_cols = [
    "day_of_week", "is_weekend", "month", "days_since_start",
    "cancel_return_rate", "discount_rate", "top_category_share",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14",
    "revenue_roll7_mean", "revenue_roll7_std", "revenue_roll30_mean",
]

test_start = feat["date"].max() - pd.Timedelta(days=55)
val_start = test_start - pd.Timedelta(days=56)

train = feat[feat["date"] < val_start]
val = feat[(feat["date"] >= val_start) & (feat["date"] < test_start)]
test = feat[feat["date"] >= test_start]
print(f"train: {len(train)} rows | val: {len(val)} rows | test: {len(test)} rows")

X_train, y_train = train[feature_cols], train["revenue"]
X_val, y_val = val[feature_cols], val["revenue"]
X_test, y_test = test[feature_cols], test["revenue"]

baseline_pred = X_test["revenue_lag_7"]
print("\nBaseline (same day last week):")
print("  MAE :", round(mean_absolute_error(y_test, baseline_pred), 2))
print("  MAPE:", round(mean_absolute_percentage_error(y_test, baseline_pred) * 100, 1), "%")

settings = [
    {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2},
    {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 2},
    {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 1},
]
val_results = []
for s in settings:
    m = RandomForestRegressor(random_state=42, **s)
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    val_results.append({**s, "val_MAE": mean_absolute_error(y_val, pred)})

val_df = pd.DataFrame(val_results)
print("\nValidation results (used only to choose settings):\n", val_df)

best = val_df.loc[val_df["val_MAE"].idxmin()]
print("\nChosen settings:", best.to_dict())

final_train = pd.concat([train, val])
final_model = RandomForestRegressor(
    random_state=42,
    n_estimators=int(best["n_estimators"]),
    max_depth=int(best["max_depth"]),
    min_samples_leaf=int(best["min_samples_leaf"]),
)
final_model.fit(final_train[feature_cols], final_train["revenue"])
pred = final_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred) * 100
print("\nFinal model - honest test performance:")
print("  MAE :", round(mae, 2))
print("  MAPE:", round(mape, 1), "%")

importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:\n", importances)

results_df = pd.DataFrame({"date": test["date"], "actual": y_test, "predicted": pred})
results_df.to_csv("clean_data/test_predictions.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(results_df["date"], results_df["actual"], label="Actual")
plt.plot(results_df["date"], results_df["predicted"], label="Predicted")
plt.title("Random Forest — Actual vs Predicted (held-out test set)")
plt.xlabel("Date"); plt.ylabel("Revenue"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

joblib.dump(final_model, "outputs/forecast_model_v2.pkl")
