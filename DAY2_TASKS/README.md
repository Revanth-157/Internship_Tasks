# Exploratory Data Analysis (EDA) – Titanic Dataset

## 📌 Objective
Perform Exploratory Data Analysis on the Titanic dataset to understand feature distributions, relationships, and detect anomalies or patterns using visualization tools.

---

## 🔧 Tools Used
- **Pandas** – Data manipulation & summary statistics  
- **NumPy** – Numerical operations  
- **Matplotlib** – Basic plotting  
- **Seaborn** – Advanced visualizations  
- **Plotly** – Interactive correlation matrix  

---

## 🗂️ Dataset
- **Source:** Titanic-Dataset.csv  
- **Target Variable:** `Survived`  
- **Feature Types:**
  - Numeric: `Age`, `Fare`, `Pclass`, etc.
  - Categorical: `Sex`, `Embarked`, etc.

---

## 📝 Mini Guide

### 1. 📊 Summary Statistics
Generate basic statistics:
- Mean, Median, Standard Deviation (std), Min/Max
```python
data.describe()
data.median(numeric_only=True)
data.mode(numeric_only=True)
