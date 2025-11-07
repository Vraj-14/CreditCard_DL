
import pandas as pd
X_test = pd.read_csv('data/processed/X_test.csv')
demo = X_test.sample(20, random_state=42).reset_index(drop=True)
demo.to_csv('data/demo_transactions.csv', index=False)