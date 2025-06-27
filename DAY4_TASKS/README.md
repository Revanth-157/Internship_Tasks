# Breast Cancer Classification using Logistic Regression

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset** to build a binary classifier using **Logistic Regression**. The primary goal is to distinguish between **malignant (M)** and **benign (B)** tumors based on various cell feature measurements.

---

## 📌 Tasks Performed

### ✅ 1. Data Preprocessing
- Loaded the dataset from `data.csv`.
- Encoded target column `diagnosis` as:
  - Malignant → `1`
  - Benign → `0`
- Scaled numeric features using `MinMaxScaler` to bring values between 0 and 1.
- Binarized features based on a threshold (`> 0.2` → `1`, else `0`).

### ✅ 2. Train/Test Split
- Removed `id` column.
- Split data into **80% training** and **20% test** using `train_test_split`.

### ✅ 3. Model Training
- Trained a **Logistic Regression** model using the training data.

### ✅ 4. Model Evaluation
- Evaluated using:
  - Confusion Matrix
  - Precision
  - Recall
  - ROC-AUC Score
  - Classification Report

### ✅ 5. Threshold Tuning
- Changed the classification threshold from default `0.5` to `0.3` to observe impact on recall and precision.

---

## 🧠 Interview Questions & Answers

### 1. **How does logistic regression differ from linear regression?**
- Linear regression predicts **continuous values**.
- Logistic regression predicts **probabilities for classification**, typically using the sigmoid function to output values between 0 and 1.

---

### 2. **What is the sigmoid function?**
- A function that maps any real-valued number to a value between 0 and 1:
  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

- It's used in logistic regression to interpret the output as a probability.

---

### 3. **What is precision vs recall?**
| Metric    | Formula                   | Meaning                                                  |
|-----------|---------------------------|-----------------------------------------------------------|
| Precision | TP / (TP + FP)            | Of predicted positives, how many are truly positive       |
| Recall    | TP / (TP + FN)            | Of actual positives, how many did the model identify      |

---

### 4. **What is the ROC-AUC curve?**
- **ROC (Receiver Operating Characteristic)** curve plots **True Positive Rate (Recall)** vs **False Positive Rate** at various thresholds.
- **AUC (Area Under Curve)** summarizes the overall performance.
- AUC = 1 → perfect, AUC = 0.5 → random guessing.

---

### 5. **What is the confusion matrix?**
A table that describes the performance of a classifier:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP)| True Negative (TN) |

---

### 6. **What happens if classes are imbalanced?**
- The model may favor the **majority class**, leading to poor recall for the minority class.
- Use techniques like **class weighting**, **oversampling**, or **evaluation metrics** like F1-score/ROC-AUC instead of accuracy.

---

### 7. **How do you choose the threshold?**
- The default is **0.5**, but you can adjust it based on:
  - Business goals (e.g., high recall in medical diagnosis).
  - ROC or Precision-Recall curves.
  - Maximize F1-score or minimize cost.

---

### 8. **Can logistic regression be used for multi-class problems?**
Yes. Logistic regression can be extended to multi-class using strategies like:
- **One-vs-Rest (OvR)**
- **Softmax Regression (Multinomial Logistic Regression)**

---

## 📁 Files

- `data.csv` – Breast Cancer dataset
- `
