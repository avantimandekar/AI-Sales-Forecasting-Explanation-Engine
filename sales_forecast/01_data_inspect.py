import pandas as pd

files = {
    'customers': 'raw_data/customers.csv',
    'orders': 'raw_data/orders.csv',
    'order_items': 'raw_data/order_items.csv',
    'products': 'raw_data/products.csv',
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print("nulls:\n", df.isnull().sum())
    print("full-row duplicates:", df.duplicated().sum())