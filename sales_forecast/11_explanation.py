import pandas as pd
import joblib
import shap

feat = pd.read_csv("clean_data/features.csv", parse_dates=["date"])
forecast = pd.read_csv("clean_data/future_forecast.csv", parse_dates=["date"])
model = joblib.load("outputs/forecast_model_v2.pkl")

feature_cols = [
    "day_of_week", "is_weekend", "month", "days_since_start",
    "cancel_return_rate", "discount_rate", "top_category_share",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14",
    "revenue_roll7_mean", "revenue_roll7_std", "revenue_roll30_mean",
]

readable_names = {
    "cancel_return_rate": "cancellation/return rate",
    "discount_rate": "discount code usage",
    "top_category_share": "category concentration",
    "revenue_lag_1": "yesterday's revenue",
    "revenue_lag_7": "revenue same day last week",
    "revenue_lag_14": "revenue 14 days ago",
    "revenue_roll7_mean": "recent 7-day sales",
    "revenue_roll7_std": "recent day-to-day volatility",
    "revenue_roll30_mean": "30-day revenue trend",
    "day_of_week": "day of week",
    "month": "month",
    "is_weekend": "weekend",
    "days_since_start": "overall trend",
}

sample = forecast.iloc[0]
sample_df = pd.DataFrame([sample[feature_cols]])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_df)
contributions = pd.Series(shap_values[0], index=feature_cols).sort_values(key=abs, ascending=False)

print("=" * 55)
print("AI SALES FORECASTING + EXPLANATION ENGINE")
print("=" * 55)
print(f"\nForecast for {sample['date'].date()}: ₹{sample['forecast_revenue']:,.0f}")

print("\nModel Explanation:")
for feature, impact in contributions.head(5).items():
    name = readable_names.get(feature, feature)
    direction = "increased" if impact > 0 else "decreased"
    print(f" - {name} {direction} the forecast by ₹{abs(impact):,.0f}")

print("\nHistorical Insights:")
hist_revenue_mean = feat["revenue_roll30_mean"].mean()
hist_discount_mean = feat["discount_rate"].mean()
hist_cancel_mean = feat["cancel_return_rate"].mean()

trend_ratio = sample["revenue_roll30_mean"] / hist_revenue_mean
if trend_ratio > 1.1:
    print(f" - 30-day revenue trend is {trend_ratio:.1f}x the historical average")
elif trend_ratio < 0.9:
    print(f" - 30-day revenue trend is only {trend_ratio:.1f}x the historical average")

if sample["revenue_roll7_mean"] < sample["revenue_roll30_mean"] * 0.85:
    print(f" - recent 7-day sales (₹{sample['revenue_roll7_mean']:,.0f}) are below "
          f"the 30-day trend (₹{sample['revenue_roll30_mean']:,.0f})")

discount_diff = sample["discount_rate"] - hist_discount_mean
if discount_diff > 0.05:
    print(f" - discount usage is {discount_diff*100:.1f} percentage points above average")

cancel_diff = sample["cancel_return_rate"] - hist_cancel_mean
if cancel_diff > 0.03:
    print(f" - cancellation/return rate is {cancel_diff*100:.1f} percentage points above average")
