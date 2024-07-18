import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import X_scaled, y, data

# Visualize the distribution of the target variable
sns.countplot(x=y)
plt.title('Distribution of Churn')
plt.show()

# Visualize correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(X_scaled.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Visualize distribution of numerical features
data.hist(bins=30, figsize=(15, 15))
plt.suptitle('Feature Distributions')
plt.show()