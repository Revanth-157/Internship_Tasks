import numpy as np
import pandas as pds
import re
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# File path
file = "Titanic-Dataset.csv"

# Task 1: Load the dataset and explore basic info
data = pds.read_csv(file)
print(data.info())
print(data.describe())
print(data.head())
print(data.tail())

# Task 2: Handle missing values (mean for numeric, mode for categorical)
for i in data.columns:
    if data[i].dtype in ["int64", "float64"]:
        data[i] = data[i].fillna(data[i].mean())
    else:
        data[i] = data[i].fillna(data[i].mode()[0])

# Task 3: Convert categorical features into numerical
for i in data.columns:
    if data[i].dtype not in ["int64", "float64"]:
        if i == "Sex":
            data[i] = data[i].replace({'male': 0, 'female': 1})
        
        if i == "Name":
            # Extract titles like Mr, Mrs, etc.
            data['Title'] = data[i].apply(lambda x: re.search(r'\b(Mr|Mrs|Miss|Master|Dr|Rev|Col|Major|Capt|Ms|Mlle|Mme|Don|Sir|Lady|Jonkheer|Countess)\b', x))
            data['Title'] = data['Title'].apply(lambda x: x.group() if x else 'Other')
            data = pds.get_dummies(data, columns=['Title'], drop_first=True)
        
        if i == "Embarked":
            data = pds.get_dummies(data, columns=['Embarked'], drop_first=True)

        if i == "Cabin":
            # Extract first letter of cabin or mark unknowns
            data[i] = data[i].apply(lambda x: re.search(r"^[A-Z]", str(x)).group() if re.search(r"^[A-Z]", str(x)) else "U")
            data = pds.get_dummies(data, columns=[i], drop_first=True)

# Drop unnecessary columns
data = data.drop(columns=["Ticket", "Name"])

# Task 4: Standardize numerical features
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Task 5: Visualize and remove outliers using IQR
for col in numeric_cols:
    plt.figure(figsize=(6, 1))
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Remove outliers using IQR
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower) & (data[col] <= upper)]

# Final info after preprocessing
print("\n✅ Final dataset after preprocessing:")
print(data.info())

        
