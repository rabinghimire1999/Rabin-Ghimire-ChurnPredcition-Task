import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('churn-bigml-80.csv')

# Remove the 'State' column
data = data.drop('State', axis=1)

# Check for missing values
print(data.isnull().sum())

# Separate features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Ensure 'Churn' is a binary categorical variable
y = y.astype(int)

# Separate numerical and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=[object]).columns

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Handling missing values for numerical columns
imputer_num = SimpleImputer(strategy='mean')
X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])

# Handling missing values for categorical columns
imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])


# Convert categorical variables to numerical format
label_encoders = {}
for column in categorical_cols:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])


# Scale the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(X_scaled.head())
print(y.head())
