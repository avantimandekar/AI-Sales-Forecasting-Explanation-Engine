import pandas as pd
import joblib

feat = pd.read_csv("clean_data/features.csv", parse_dates=["date"])
model = joblib.load("outputs/forecast_model_v2.pkl")

feature_cols = [
    "day_of_week", "is_weekend", "month", "days_since_start",
    "cancel_return_rate", "discount_rate", "top_category_share",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14",
    "revenue_roll7_mean", "revenue_roll7_std", "revenue_roll30_mean",
]

N_DAYS_AHEAD = 30
recent_30 = feat.tail(30)
assumed_cancel_return_rate = recent_30["cancel_return_rate"].mean()
assumed_discount_rate = recent_30["discount_rate"].mean()
assumed_top_category_share = recent_30["top_category_share"].mean()

revenue_history = feat[["date", "revenue"]].copy()
start_date = feat["date"].max()
days_since_start_0 = feat["days_since_start"].max()

forecast_rows = []
for step in range(1, N_DAYS_AHEAD + 1):
    future_date = start_date + pd.Timedelta(days=step)
    lag1 = revenue_history["revenue"].iloc[-1]
    lag7 = revenue_history["revenue"].iloc[-7]
    lag14 = revenue_history["revenue"].iloc[-14]
    roll7 = revenue_history["revenue"].tail(7)
    roll30 = revenue_history["revenue"].tail(30)

    row = {
        "date": future_date,
        "day_of_week": future_date.dayofweek,
        "is_weekend": int(future_date.dayofweek >= 5),
        "month": future_date.month,
        "days_since_start": days_since_start_0 + step,
        "cancel_return_rate": assumed_cancel_return_rate,
        "discount_rate": assumed_discount_rate,
        "top_category_share": assumed_top_category_share,
        "revenue_lag_1": lag1, "revenue_lag_7": lag7, "revenue_lag_14": lag14,
        "revenue_roll7_mean": roll7.mean(), "revenue_roll7_std": roll7.std(),
        "revenue_roll30_mean": roll30.mean(),
    }
    X_future = pd.DataFrame([row])[feature_cols]
    pred = max(0, model.predict(X_future)[0])
    row["forecast_revenue"] = pred
    forecast_rows.append(row)
    revenue_history = pd.concat([revenue_history, pd.DataFrame([{"date": future_date, "revenue": pred}])], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv("clean_data/future_forecast.csv", index=False)

print("=" * 55)
print("AI SALES FORECASTING SYSTEM")
print("=" * 55)
print(f"\nForecast for {forecast_df.iloc[0]['date'].date()}: ₹{forecast_df.iloc[0]['forecast_revenue']:,.0f}")
print("\nNext 7 days:")
print(forecast_df[["date", "forecast_revenue"]].head(7).to_string(index=False))