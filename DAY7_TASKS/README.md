# 🧠 Task 7: Support Vector Machines (SVM)

This project demonstrates the use of Support Vector Machines (SVM) for binary classification using the **Breast Cancer Dataset**. The objective is to understand and implement both **linear** and **non-linear (RBF)** SVMs, visualize decision boundaries, perform **hyperparameter tuning**, and evaluate model performance using **cross-validation**.

---

## ✅ Mini Guide

### 1. Load and Prepare a Dataset for Binary Classification
- The breast cancer dataset is used.
- Preprocessing includes:
  - Dropping unnecessary columns (e.g., `id`)
  - Mapping categorical values (`M` to 1, `B` to 0)
  - Scaling numeric features using `StandardScaler`

### 2. Train an SVM with Linear and RBF Kernel
- Trained two models:
  - `SVC(kernel='linear')`
  - `SVC(kernel='rbf')`

### 3. Visualize Decision Boundary Using 2D Data
- Two features (`radius_mean` and `texture_mean`) are selected.
- Trained linear and RBF SVMs.
- Used `matplotlib` to plot decision boundaries with colored regions.

### 4. Tune Hyperparameters like `C` and `gamma`
- `GridSearchCV` is used with a 5-fold cross-validation.
- Best `C` and `gamma` parameters are selected for the RBF kernel.

### 5. Use Cross-Validation to Evaluate Performance
- Used `cross_val_score` with `cv=5` to evaluate SVMs with:
  - Linear Kernel
  - Polynomial Kernel
  - RBF Kernel

---

## 📌 Interview Questions

1. **What is a support vector?**  
   A support vector is a data point that lies closest to the decision boundary (hyperplane) and helps define its position and orientation.

2. **What does the `C` parameter do?**  
   The `C` parameter controls the trade-off between achieving a low error on training data and minimizing the margin. Lower `C` values lead to a wider margin and more tolerance to misclassification.

3. **What are kernels in SVM?**  
   Kernels are functions that transform the input data into higher-dimensional space to make it possible to perform linear separation in that space.

4. **What is the difference between linear and RBF kernel?**  
   - **Linear kernel:** Suitable for linearly separable data.
   - **RBF (Radial Basis Function):** Projects data into higher dimensions, useful for non-linearly separable data.

5. **What are the advantages of SVM?**  
   - Effective in high-dimensional spaces.
   - Works well for both linearly and non-linearly separable data using kernels.
   - Robust against overfitting, especially in low-dimensional feature spaces.

6. **Can SVMs be used for regression?**  
   Yes, SVMs can be used for regression using a variant called **SVR** (Support Vector Regression).

7. **What happens when data is not linearly separable?**  
   The SVM uses a **soft margin** and kernel tricks (like RBF) to separate the data in a transformed space.

8. **How is overfitting handled in SVM?**  
   Overfitting is controlled by:
   - The `C` parameter (regularization)
   - Choosing the appropriate kernel
   - Cross-validation and hyperparameter tuning

---

## 📂 Dataset

- **Dataset Used**: Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Download Link**: [Click here to download dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## 🔧 Requirements

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

Install via pip:
```bash
pip install scikit-learn pandas numpy matplotlib
