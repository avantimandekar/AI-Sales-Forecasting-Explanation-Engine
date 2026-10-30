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

daily = completed.groupby("date").agg(
    revenue=("revenue", "sum"),
    units_sold=("quantity", "sum"),
    order_count=("order_id", "nunique")
).reset_index()

all_daily = full.groupby("date").agg(
    total_orders=("order_id", "nunique")
).reset_index()

cancel_return = full[
    full["status"].isin(["canceled", "returned"])
].groupby("date").agg(
    cancel_return_orders=("order_id", "nunique")
).reset_index()

all_daily = all_daily.merge(
    cancel_return,
    on="date",
    how="left"
)

all_daily["cancel_return_orders"] = (
    all_daily["cancel_return_orders"].fillna(0)
)

all_daily["cancel_return_rate"] = (
    all_daily["cancel_return_orders"]
    / all_daily["total_orders"]
)

daily = daily.merge(
    all_daily[["date", "cancel_return_rate"]],
    on="date",
    how="left"
)

daily = daily.sort_values("date").reset_index(drop=True)

daily.to_csv(
    "clean_data/main.csv",
    index=False
)

print("Shape:", daily.shape)
print("Start date:", daily["date"].min())
print("End date:", daily["date"].max())
print(daily.head())