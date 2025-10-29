import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# Load test data
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Load models
with open('models/logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)
with open('models/xgboost_model.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)

# Evaluate
results = []
for name, model in [('Logistic Regression', logistic_model), ('XGBoost', xgboost_model)]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'Model': name,
        'Precision (Fraud)': report['1']['precision'],
        'Recall (Fraud)': report['1']['recall'],
        'F1-Score (Fraud)': report['1']['f1-score'],
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

# Save results
pd.DataFrame(results).to_csv('outputs/results/model_metrics.csv', index=False)
print(pd.DataFrame(results))