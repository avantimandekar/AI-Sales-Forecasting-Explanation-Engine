import pandas as pd

customers = pd.read_csv(
    "clean_data/clean_customers.csv",
    parse_dates=["signup_date"]
)

orders = pd.read_csv(
    "clean_data/clean_orders.csv",
    parse_dates=["order_date"]
)

items = pd.read_csv("clean_data/clean_order_items.csv")
products = pd.read_csv("clean_data/clean_products.csv")

daily = pd.read_csv(
    "clean_data/main.csv",
    parse_dates=["date"]
)

full = items.merge(
    orders,
    on="order_id",
    how="left"
)

full = full.merge(
    products,
    on="product_id",
    how="left"
)

full["date"] = full["order_date"].dt.floor("D")

completed = full[
    full["status"] == "completed"
].copy()

orders["date"] = orders["order_date"].dt.floor("D")

discount_rate = (
    orders.groupby("date")["discount_code"]
    .apply(lambda x: (x != "NO_DISCOUNT").mean())
    .reset_index(name="discount_rate")
)

category_revenue = (
    completed.groupby(["date", "category"])["revenue"]
    .sum()
    .reset_index()
)

top_category = (
    category_revenue.groupby("date")["revenue"]
    .max()
    .reset_index(name="top_category_revenue")
)

total_category_revenue = (
    category_revenue.groupby("date")["revenue"]
    .sum()
    .reset_index(name="total_category_revenue")
)

top_category_share = top_category.merge(
    total_category_revenue,
    on="date"
)

top_category_share["top_category_share"] = (
    top_category_share["top_category_revenue"]
    / top_category_share["total_category_revenue"]
)

daily = daily.merge(
    discount_rate,
    on="date",
    how="left"
)

daily = daily.merge(
    top_category_share[["date", "top_category_share"]],
    on="date",
    how="left"
)

daily = daily.sort_values("date").reset_index(drop=True)

daily["cancel_return_rate"] = (
    daily["cancel_return_rate"].shift(1)
)

daily["discount_rate"] = (
    daily["discount_rate"].shift(1)
)

daily["top_category_share"] = (
    daily["top_category_share"].shift(1)
)

daily["day_of_week"] = daily["date"].dt.dayofweek

daily["is_weekend"] = (
    daily["day_of_week"] >= 5
).astype(int)

daily["month"] = daily["date"].dt.month

daily["days_since_start"] = (
    daily["date"] - daily["date"].min()
).dt.days

daily["revenue_lag_1"] = daily["revenue"].shift(1)
daily["revenue_lag_7"] = daily["revenue"].shift(7)
daily["revenue_lag_14"] = daily["revenue"].shift(14)

daily["revenue_roll7_mean"] = (
    daily["revenue"].shift(1).rolling(7).mean()
)

daily["revenue_roll7_std"] = (
    daily["revenue"].shift(1).rolling(7).std()
)

daily["revenue_roll30_mean"] = (
    daily["revenue"].shift(1).rolling(30).mean()
)

before = len(daily)

features = daily.dropna().reset_index(drop=True)

print(
    f"Dropped {before - len(features)} rows. "
    f"{len(features)} rows remain."
)

features.to_csv(
    "clean_data/features.csv",
    index=False
)

print("Shape:", features.shape)
print(features.head())