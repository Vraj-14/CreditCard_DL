import pandas as pd

# Load original data
df = pd.read_csv('dataset/creditcard.csv')

# Simple feature: Time difference (seconds since first transaction)
df['Time_diff'] = df['Time'] - df['Time'].min()

# Scale new feature
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Time_diff_scaled'] = scaler.fit_transform(df[['Time_diff']])
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

# Drop original columns
df = df.drop(['Time', 'Amount', 'Time_diff'], axis=1)

# Save updated data
X = df.drop('Class', axis=1)
y = df['Class']
X.to_csv('data/processed/X_features.csv', index=False)
y.to_csv('data/processed/y_features.csv', index=False)