# create_demo_with_fraud.py
import pandas as pd

# Load processed test data
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Combine
test_df = pd.concat([X_test, y_test], axis=1)

# Pick 10 legit + 10 fraud
legit = test_df[test_df['Class'] == 0].sample(10, random_state=42)
fraud = test_df[test_df['Class'] == 1].sample(10, random_state=42)

# Combine
demo = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

# Drop Class (not needed in UI)
demo = demo.drop('Class', axis=1)

# Save
demo.to_csv('data/demo_transactions.csv', index=False)
print("Demo updated with 10 fraud + 10 legit transactions")
print("Fraud indices in demo:", demo.index[demo.index >= 10].tolist())