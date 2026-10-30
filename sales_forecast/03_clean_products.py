import pandas as pd

products = pd.read_csv("raw_data/products.csv")

print("Shape:", products.shape)


print(products.isnull().sum())


print("Duplicate rows:", products.duplicated().sum())


for column in products.columns:
    print(f"\n{column}")
    print("Unique count:", products[column].nunique(dropna=False))
    print(products[column].unique())

products["category"] = products["category"].str.strip().str.title()
print(products["category"].value_counts())
print("Unique categories:", products["category"].nunique())


print(products[["list_price", "unit_cost"]].dtypes)

print("List price <= 0:", (products["list_price"] <= 0).sum())
print("Unit cost <= 0:", (products["unit_cost"] <= 0).sum())


print(
    "Cost > list price:",
    (products["unit_cost"] > products["list_price"]).sum()
)

print("Final Shape:", products.shape)
products.to_csv('clean_data/clean_products.csv', index=False)