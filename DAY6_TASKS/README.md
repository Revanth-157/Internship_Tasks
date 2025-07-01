# 📊 K-Nearest Neighbors (KNN) Classifier Project

## 🎯 Objective
To implement and explore the **K-Nearest Neighbors (KNN)** algorithm for classification using the **Iris dataset**. This includes data preprocessing, experimenting with different values of K, evaluating model performance, and visualizing decision boundaries.

---

## 🛠 Tools & Libraries
- Python 3.x  
- Scikit-learn  
- Pandas  
- Matplotlib  
- Seaborn  
- NumPy  

---

## 📂 Dataset
**File**: `Iris.csv`  
**Target column**: `species`  
**Features**: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`  

---

## 📘 Mini Guide: Task Breakdown

### ✅ 1. Load Dataset and Normalize Features
- Load `Iris.csv` into a DataFrame using `pandas`.
- Drop unnecessary columns like `Id` if present.
- Encode the `species` column using `LabelEncoder` to convert string labels to integers.
- Normalize the features using `StandardScaler` to ensure all features contribute equally to distance calculations.

### ✅ 2. Use `KNeighborsClassifier` from sklearn
- Split the dataset into training and test sets using `train_test_split`.
- Train a KNN model using `KNeighborsClassifier(n_neighbors=3)`.

### ✅ 3. Experiment with Different Values of K
- Loop through values of K (1 to 10).
- Train a new model for each K and evaluate the accuracy on the test set.
- Choose the K with the best performance.

### ✅ 4. Evaluate Model Using Accuracy and Confusion Matrix
- Use `accuracy_score` to evaluate the model.
- Use `confusion_matrix` to visualize how well the model performs on each class.
- Display the matrix using `seaborn.heatmap`.

### ✅ 5. Visualize Decision Boundaries
- Reduce feature dimensions using PCA to 2 components.
- Re-train KNN on the reduced dataset.
- Use meshgrid and `contourf()` to visualize the decision regions and actual data points.

---

## 🧠 Interview Questions & Answers

### 1. How does the KNN algorithm work?
KNN classifies a data point based on the majority class of its **K nearest neighbors** in the feature space using a distance metric like Euclidean distance.

### 2. How do you choose the right K?
Use **cross-validation** or **error-rate vs K plot**; odd values help avoid ties. Choose the K that minimizes error or maximizes validation accuracy.

### 3. Why is normalization important in KNN?
KNN uses distance metrics. If features are not scaled, larger-valued features dominate distance calculations, leading to poor predictions.

### 4. What is the time complexity of KNN?
- **Training Time**: O(1) (KNN is a lazy learner)
- **Prediction Time**: O(n × d), where *n* = training samples and *d* = number of features

### 5. What are the pros and cons of KNN?

| Pros                         | Cons                                  |
|------------------------------|----------------------------------------|
| Simple and intuitive         | Slow for large datasets                |
| No training phase            | High memory usage                      |
| Naturally supports multi-class | Sensitive to noise and irrelevant features |

### 6. Is KNN sensitive to noise?
Yes. Noisy or outlier data can heavily influence the prediction because KNN considers local data points only.

### 7. How does KNN handle multi-class problems?
KNN naturally supports multi-class classification by **majority vote** among the K neighbors.

### 8. What’s the role of distance metrics in KNN?
Distance metrics (like **Euclidean**, **Manhattan**, **Minkowski**) define how closeness is measured between points. The choice of metric significantly affects predictions.

---

## ✅ Summary
This project demonstrates how to apply the KNN algorithm from scratch using a real-world dataset. It includes:
- Data loading and preprocessing  
- K tuning  
- Model evaluation  
- Decision boundary visualization  

Explore different datasets, scalers, and distance metrics to gain deeper insights into how KNN behaves in various settings.
