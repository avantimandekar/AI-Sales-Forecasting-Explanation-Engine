import streamlit as st
import pandas as pd
import joblib
import shap
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Sales Forecasting", layout="centered")

feature = pd.read_csv("clean_data/features.csv", parse_dates=["date"])
backtest = pd.read_csv("clean_data/test_predictions.csv", parse_dates=["date"])
main = pd.read_csv("clean_data/main.csv", parse_dates=["date"])
future = pd.read_csv("clean_data/future_forecast.csv", parse_dates=["date"])
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

explainer = shap.TreeExplainer(model)

def get_shap_contributions(row_df):
    shap_values = explainer.shap_values(row_df)
    values = shap_values[0] if isinstance(shap_values, list) else shap_values
    if hasattr(values, "ndim") and values.ndim > 1:
        values = values[0]
    return pd.Series(values, index=feature_cols).sort_values(key=abs, ascending=False)

def show_contributions(contributions, top_n=5):
    for feature, impact in contributions.head(top_n).items():
        name = readable_names.get(feature, feature)
        if impact > 0:
            st.write(f"↑ {name} increased the prediction by ₹{abs(impact):,.0f}")
        else:
            st.write(f"↓ {name} decreased the prediction by ₹{abs(impact):,.0f}")

st.title("AI Sales Forecasting System")
st.caption("Ecommerce revenue forecasting using machine learning")

# ---------------------------------------------------------------
# SECTION 1: Backtest - Prediction vs Actual on a chosen past date
# ---------------------------------------------------------------
feat_backtest = feature.merge(backtest[["date", "predicted"]], on="date", how="inner")
test_dates = feat_backtest["date"].dt.date.tolist()

selected_date = st.selectbox("Select a date", test_dates, index=len(test_dates) - 1)
row = feat_backtest[feat_backtest["date"].dt.date == selected_date].iloc[0]
row_df = pd.DataFrame([row[feature_cols]])

prediction = row["predicted"]
actual = row["revenue"]
absolute_error = abs(actual - prediction)

st.subheader("Prediction vs Actual")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Revenue", f"₹{prediction:,.0f}")
col2.metric("Actual Revenue", f"₹{actual:,.0f}")
col3.metric("Absolute Error", f"₹{absolute_error:,.0f}")

# ---------------------------------------------------------------
# SECTION 2: Model Performance
# ---------------------------------------------------------------
st.subheader("Model Performance")

baseline_data = backtest.merge(feature[["date", "revenue_lag_7"]], on="date", how="left")
baseline_mae = mean_absolute_error(baseline_data["actual"], baseline_data["revenue_lag_7"])
model_mae = mean_absolute_error(backtest["actual"], backtest["predicted"])
mape = (abs((backtest["actual"] - backtest["predicted"]) / backtest["actual"]).mean()) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Baseline MAE", f"₹{baseline_mae:,.0f}")
col2.metric("Model MAE", f"₹{model_mae:,.0f}")
col3.metric("Model MAPE", f"{mape:.1f}%")

# ---------------------------------------------------------------
# SECTION 3: SHAP explanation for the selected backtest date
# ---------------------------------------------------------------
st.subheader("Model Explanation")
show_contributions(get_shap_contributions(row_df))

# ---------------------------------------------------------------
# SECTION 4: Business-context, rule-based insights
# ---------------------------------------------------------------
st.subheader("Historical Insights")

hist_revenue_mean = feature["revenue_roll30_mean"].mean()
hist_discount_mean = feature["discount_rate"].mean()
hist_cancel_mean = feature["cancel_return_rate"].mean()

trend_ratio = row["revenue_roll30_mean"] / hist_revenue_mean
if trend_ratio > 1.1 or trend_ratio < 0.9:
    st.write(f"• 30-day revenue trend is {trend_ratio:.1f}x the historical average")

if row["revenue_roll7_mean"] < row["revenue_roll30_mean"] * 0.85:
    st.write(f"• Recent 7-day sales (₹{row['revenue_roll7_mean']:,.0f}) are below "
             f"the 30-day trend (₹{row['revenue_roll30_mean']:,.0f})")

discount_diff = row["discount_rate"] - hist_discount_mean
if discount_diff > 0.05:
    st.write(f"• Discount usage is {discount_diff * 100:.1f} percentage points above average")

cancel_diff = row["cancel_return_rate"] - hist_cancel_mean
if cancel_diff > 0.03:
    st.write(f"• Cancellation/return rate is {cancel_diff * 100:.1f} percentage points above average")

# ---------------------------------------------------------------
# SECTION 5: 30-day future forecast + explanation for a chosen future date
# ---------------------------------------------------------------
st.subheader("30-Day Future Forecast")

total_forecast = future["forecast_revenue"].sum()
average_forecast = future["forecast_revenue"].mean()

col1, col2 = st.columns(2)
col1.metric("Total Forecasted Revenue (30 days)", f"₹{total_forecast:,.0f}")
col2.metric("Average Daily Forecast", f"₹{average_forecast:,.0f}")

st.line_chart(future.set_index("date")["forecast_revenue"])

st.subheader("Explain a Future Date")
future_dates = future["date"].dt.date.tolist()
selected_future_date = st.selectbox("Select a forecast date", future_dates)

future_row = future[future["date"].dt.date == selected_future_date].iloc[0]
future_row_df = pd.DataFrame([future_row[feature_cols]])

st.write(f"**Forecast: ₹{future_row['forecast_revenue']:,.0f}**")
show_contributions(get_shap_contributions(future_row_df))

st.caption("Note: discount rate, cancellation rate, and category mix are assumed to stay "
           "at their recent 30-day average for future dates, since actual future orders "
           "aren't known yet.")

# ---------------------------------------------------------------
# SECTION 6: Full history, for context
# ---------------------------------------------------------------
st.subheader("Recent Revenue Trend")
st.line_chart(main.set_index("date")["revenue"].tail(90))