import pandas as pd

orders = pd.read_csv("raw_data/orders.csv")

print("Shape:", orders.shape)


print(orders.isnull().sum())


print("Duplicate rows:", orders.duplicated().sum())


for column in orders.columns:
    print(f"\n{column}")
    print("Unique count:", orders[column].nunique(dropna=False))
    print(orders[column].unique())


print("Total rows:", len(orders))
print("Unique order IDs:", orders["order_id"].nunique())
print("Duplicate order IDs:", orders["order_id"].duplicated().sum())



duplicates = orders[orders.duplicated(keep=False)].sort_values("order_id")
print(duplicates)

before = len(orders)
orders = orders.drop_duplicates()
print(f"Removed {before - len(orders)} exact duplicate rows")


orders['order_date'] = pd.to_datetime(orders['order_date'], errors="coerce")

print("\nMissing dates after conversion:")
print(orders["order_date"].isna().sum())

print("\nDate range:")
print(orders["order_date"].min())
print(orders["order_date"].max())


print(orders["discount_code"].value_counts(dropna=False))
orders["discount_code"] = orders["discount_code"].fillna("NO_DISCOUNT")


print(orders["status"].value_counts())
print(orders["payment_method"].value_counts())

print("Final Shape:", orders.shape)
orders.to_csv('clean_data/clean_orders.csv', index=False)
