# LOGISTIC REGRESSION

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import pickle
import matplotlib.pyplot as plt

# Load SMOTE-balanced training data and test data
X_train = pd.read_csv('data/processed/X_train_smote.csv')
y_train = pd.read_csv('data/processed/y_train_smote.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Train logistic regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train.values.ravel())

# Predictions and evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("Logistic Regression - Classification Report:\n", classification_report(y_test, y_pred))
print("Logistic Regression - ROC-AUC:", roc_auc_score(y_test, y_proba))

# Save model
with open('models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend()
plt.savefig('outputs/plots/roc_logistic.png')
plt.close()