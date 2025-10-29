# GENERATES class_distribution.png & amount_distribution.png

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load dataset
# df = pd.read_csv('dataset/creditcard.csv')

# # Basic checks
# print("Dataset Shape:", df.shape)  # Should be ~284,807 rows, 31 columns
# print("Missing Values:\n", df.isnull().sum())  # Should be 0
# print("Class Distribution:\n", df['Class'].value_counts(normalize=True))  # Fraud vs non-fraud ratio
# print("Basic Stats:\n", df[['Time', 'Amount']].describe())

# # Visualize Amount distribution
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='Amount', hue='Class', bins=50, log_scale=True)
# plt.title('Transaction Amount Distribution by Class')
# plt.savefig('outputs/plots/amount_distribution.png')
# plt.close()

# # Visualize fraud vs non-fraud counts
# plt.figure(figsize=(6, 4))
# sns.countplot(x='Class', data=df)
# plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
# plt.savefig('outputs/plots/class_distribution.png')
# plt.close()



# -----------------------------------------------------------------------------------------------------------

# GENERATES correlation_heatmap.png & amount_boxplot.png


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('dataset/creditcard.csv')

# Correlation heatmap (focus on V1-V28, Time_scaled, Amount_scaled)
plt.figure(figsize=(12, 8))
corr = df.drop('Class', axis=1).corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.savefig('outputs/plots/correlation_heatmap.png')
plt.close()

# Boxplot for key features by class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Amount by Class (0: Non-Fraud, 1: Fraud)')
plt.yscale('log')
plt.savefig('outputs/plots/amount_boxplot.png')
plt.close()