import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# Load data (use original, not SMOTE, for unsupervised learning)
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Filter non-fraud for training
X_train_normal = X_train[y_train['Class'] == 0]

# Define autoencoder
input_dim = X_train.shape[1]
model = Sequential([
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_normal, X_train_normal, epochs=10, batch_size=32, verbose=1)

# Evaluate reconstruction error
X_test_pred = model.predict(X_test)
mse = np.mean(np.square(X_test - X_test_pred), axis=1)
precision, recall, thresholds = precision_recall_curve(y_test, mse)
roc_auc = roc_auc_score(y_test, mse)
print("Autoencoder - ROC-AUC:", roc_auc)

# Save model
model.save('models/autoencoder.h5')

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Autoencoder Precision-Recall Curve')
plt.legend()
plt.savefig('outputs/plots/pr_autoencoder.png')
plt.close()