import pandas as pd
import matplotlib.pyplot as plt

daily = pd.read_csv('clean_data/main.csv', parse_dates=['date'])

# 1. Overall trend
daily.plot(x='date', y='revenue', title='Daily revenue over time')
plt.savefig('outputs/revenue_trend.png'); plt.close()

# 2. Day-of-week pattern
daily['day_of_week'] = daily['date'].dt.day_name()
daily.groupby('day_of_week')['revenue'].mean().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
).plot(kind='bar', title='Avg revenue by day of week')
plt.savefig('outputs/dow_pattern.png'); plt.close()

# 3. Monthly seasonality
daily['month'] = daily['date'].dt.month
daily.groupby('month')['revenue'].mean().plot(kind='bar', title='Avg revenue by month')
plt.savefig('outputs/monthly_seasonality.png'); plt.close()

print(daily['revenue'].describe())