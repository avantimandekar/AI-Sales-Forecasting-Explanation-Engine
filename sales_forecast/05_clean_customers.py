import pandas as pd

customers = pd.read_csv("raw_data/customers.csv")

print("Shape:", customers.shape)


print(customers.isnull().sum())


print("Duplicate rows:", customers.duplicated().sum())


for column in customers.columns:
    print(f"\n{column}")
    print("Unique count:", customers[column].nunique(dropna=False))
    print(customers[column].unique())


print("Duplicate customer IDs:", customers["customer_id"].duplicated().sum())
print("Customer ID <= 0:", (customers["customer_id"] <= 0).sum())


customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

print("Missing dates:", customers["signup_date"].isnull().sum())
print("Earliest signup:", customers["signup_date"].min())
print("Latest signup:", customers["signup_date"].max())



print(customers["state"].value_counts())


print(customers.isnull().sum())
print("Duplicate rows:", customers.duplicated().sum())
print("Final Shape:", customers.shape)


customers.to_csv('clean_data/clean_customers.csv', index=False)