# 🧪 K-Nearest Neighbors (KNN) Classifier - Mini Project

## 🎯 Objective
To understand and implement the **K-Nearest Neighbors (KNN)** algorithm for classification problems using the Iris dataset. Learn to normalize features, tune K, evaluate the model, and visualize decision boundaries.

---

## 🧰 Tools & Libraries
- Python 3.x  
- [Scikit-learn](https://scikit-learn.org)  
- Pandas  
- Matplotlib  
- Seaborn (optional)

---

## 📂 Dataset
- **Name**: [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
- **Features**: 4 numeric (sepal length, sepal width, petal length, petal width)  
- **Target**: 3 classes of iris flowers (Setosa, Versicolor, Virginica)

---

## 📘 Mini Guide

### ✅ 1. Load Dataset & Normalize Features
```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
