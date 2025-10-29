import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load preprocessed data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verify new distribution
print("Original Train Fraud Ratio:", y_train.mean().values[0])
print("SMOTE Train Fraud Ratio:", y_train_smote.mean().values[0])

# Save balanced data
X_train_smote.to_csv('data/processed/X_train_smote.csv', index=False)
y_train_smote.to_csv('data/processed/y_train_smote.csv', index=False)