import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import pickle
import matplotlib.pyplot as plt

# Load SMOTE-balanced training data and test data
X_train = pd.read_csv('data/processed/X_train_smote.csv')
y_train = pd.read_csv('data/processed/y_train_smote.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Train XGBoost
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train, y_train.values.ravel())

# Predictions and evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("XGBoost - Classification Report:\n", classification_report(y_test, y_pred))
print("XGBoost - ROC-AUC:", roc_auc_score(y_test, y_proba))

# Save model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend()
plt.savefig('outputs/plots/roc_xgboost.png')
plt.close()

# Feature importance
plt.figure(figsize=(10, 6))
feat_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feat_importance.plot(kind='bar')
plt.title('XGBoost Feature Importance')
plt.savefig('outputs/plots/xgboost_feature_importance.png')
plt.close()