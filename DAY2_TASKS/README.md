# Exploratory Data Analysis (EDA) – Titanic Dataset

## 📌 Objective
Perform Exploratory Data Analysis on the Titanic dataset to understand feature distributions, relationships, and detect anomalies or patterns using visualization tools.

## 🔧 Tools Used
- Pandas – Data manipulation & summary statistics  
- NumPy – Numerical operations  
- Matplotlib – Basic plotting  
- Seaborn – Advanced visualizations  
- Plotly – Interactive visualizations

## 🗂️ Dataset
- Source: Titanic-Dataset.csv  
- Target Variable: `Survived`  
- Feature Types:  
  - Numeric: `Age`, `Fare`, `Pclass`, etc.  
  - Categorical: `Sex`, `Embarked`, etc.

## 📝 Mini Guide

### 1. 📊 Generate Summary Statistics
- Use `data.describe()` for mean, std, min, max  
- Use `data.median()` and `data.mode()` for additional insights

### 2. 📈 Create Histograms and Boxplots
- Histograms with `sns.histplot()` to see distribution
- Boxplots with `sns.boxplot()` to detect outliers

### 3. 🔗 Use Pairplot / Correlation Matrix
- `sns.heatmap(data.corr())` to view correlation strength  
- `sns.pairplot()` to visually compare numeric features  
- `plotly.express.imshow()` for interactive matrix

### 4. ⚠️ Identify Patterns, Trends, or Anomalies
- Look for skewness or unusual gaps in histograms  
- Use boxplots to find extreme outliers  
- Check for missing values with `data.isnull().sum()` or heatmaps

### 5. 📉 Make Basic Feature-Level Inferences
- Use barplots (`sns.barplot`) to compare survival by category  
- Use boxplots to compare numeric features by survival outcome  
- Observe which features might influence the target (`Survived`)

## 📦 Output
- Visual analysis of feature distributions  
- Identification of key correlations and outliers  
- Foundational understanding for future modeling

## ✅ Conclusion
This EDA provides meaningful insights into the Titanic dataset, helping guide further steps in preprocessing, feature engineering, and predictive modeling.

