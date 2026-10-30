import pandas as pd

items = pd.read_csv("raw_data/order_items.csv")

print("Shape:", items.shape)


print(items.isnull().sum())


print("Duplicate rows:", items.duplicated().sum())


for column in items.columns:
    print(f"\n{column}")
    print("Unique count:", items[column].nunique(dropna=False))
    print(items[column].unique())


print("Quantity <= 0:", (items["quantity"] <= 0).sum())
print("Unit price <= 0:", (items["unit_price"] <= 0).sum())
print("Discount amount < 0:", (items["discount_amount"] < 0).sum())


products = pd.read_csv('clean_data/clean_products.csv')

before_null = items['unit_price'].isnull().sum()
items = items.merge(products[['product_id', 'list_price']], on='product_id', how='left')
items['unit_price'] = items['unit_price'].fillna(items['list_price'])
items = items.drop(columns=['list_price'])


items["revenue"] = ( items["quantity"] * items["unit_price"] - items["discount_amount"] )


print("Final Shape:", items.shape)
items.to_csv('clean_data/clean_order_items.csv', index=False)