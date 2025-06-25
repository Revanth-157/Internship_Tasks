import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
import plotly
file="Titanic-Dataset.csv"
data=pds.read_csv(file)
#Task 1: Generate summary statistics(Mean,median,sd,)
print(data.describe())
for i in data.columns:
    if data[i].dtype=="float64" or data[i].dtype=="int64":
        print("Median is",data[i].median())
        print("Mode is",data[i].mode())
#Task 2: Generate histograms and barplots for numeric features


for i in data.columns:
    if data[i].dtype in ["float64", "int64"]:
        plt.figure(figsize=(10, 4))
        plt.title(f"Histogram of {i}")
        # Fare (continuous)
        if i == "Fare":
            sns.histplot(data[i], kde=True, binwidth=30, color='skyblue')

        # Survived (binary: 0 or 1)
        elif i == "Survived":
            sns.histplot(data[i], kde=False, bins=[0,1], shrink=0.5, color='lightgreen',binwidth=0.1)

        # Pclass (categorical: 1, 2, 3)
        elif i == "Pclass":
            sns.histplot(data[i], kde=False, bins=[0.5, 1.5, 2.5, 3.5], shrink=0.5, color='lightcoral')

        # Other numeric columns
        else:
            sns.histplot(data[i], kde=True)
        plt.show()

#Task 3:-Use pairplot/correlation matrix for feature relationships.
#Correlation matrix
corr_matrix=data.corr(numeric_only=True)
#Plotting heatmap:-
plt.figure(figsize=(10,8))
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
selected = ['Age', 'Fare', 'Pclass', 'Survived']  # Example subset

sns.pairplot(data[selected], hue='Survived')  # hue can show class color
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()


#For an interactive Correlation matrix using Plotly
import plotly.express as px

# Calculate correlation matrix
corr_matrix = data.corr(numeric_only=True).round(2)

# Plot using Plotly heatmap
fig = px.imshow(corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix (Interactive)')
fig.show()
#Task 4: Identify patterns, trends, or anomalies in the data.

print(data.describe())
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

for col in numeric_cols:
    plt.figure(figsize=(8, 3))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
for col in numeric_cols:
    plt.figure(figsize=(8, 3))
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()
print(data.isnull().sum())
sns.heatmap(data.isnull(), cbar=False, cmap='Reds')
plt.title("Missing Data Heatmap")
plt.show()
#Task 5:-  .Make basic feature-level inferences from visuals.

sns.barplot(x='Pclass', y='Survived', data=data)
plt.title("Survival Rate by Passenger Class")
plt.show()
sns.boxplot(x='Survived', y='Age', data=data)
plt.title("Age Distribution by Survival")
plt.show()
sns.countplot(x='Survived', data=data)
plt.title("Class Balance: Survived vs Not")
plt.show()
