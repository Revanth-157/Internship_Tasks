# Linear Regression Model: Housing Price Prediction

## Overview

This project demonstrates the complete workflow for building a **Linear Regression Model** to predict house prices using a dataset (`Housing.csv`). It includes data preprocessing, model training, evaluation, and result visualization.

---

## Tasks and Tools

### ✅ Task 1: Import and Preprocess the Dataset

* **Tools Used**:

  * `pandas` for loading and manipulating data.
  * `SimpleImputer` from `sklearn.impute` for handling missing values.
  * `OneHotEncoder` from `sklearn.preprocessing` for categorical encoding.
  * `StandardScaler` for feature normalization.

**Steps:**

* Import dataset using `pandas.read_csv()`
* Fill missing values:

  * Numeric columns: with mean
  * Categorical columns: with mode
* Encode categorical variables using OneHotEncoding (drop first to avoid dummy variable trap)
* Normalize numerical features (optional but helpful)

### ✅ Task 2: Split Data into Train-Test Sets

* **Tool Used**: `train_test_split` from `sklearn.model_selection`
* Split feature matrix `X` and target variable `y` into training and testing sets (80/20 split).

### ✅ Task 3: Fit a Linear Regression Model

* **Tool Used**: `LinearRegression` from `sklearn.linear_model`
* Train the model using `.fit(X_train, y_train)` and predict using `.predict(X_test)`

### ✅ Task 4: Evaluate the Model

* **Tools Used**:

  * `mean_absolute_error`, `mean_squared_error`, `r2_score` from `sklearn.metrics`
* Metrics used:

  * **MAE**: Average absolute difference
  * **MSE**: Average squared difference
  * **R² Score**: Goodness of fit (1.0 = perfect prediction)

### ✅ Task 5: Plot Regression Line and Interpret Coefficients

* **Tools Used**: `matplotlib.pyplot`
* For simple regression: plot actual vs predicted values
* For multiple regression: print model coefficients using `.coef_` and `.intercept_`

---

## Interview Questions and Answers

### 1. What assumptions does linear regression make?

* Linearity, Independence of errors, Homoscedasticity (constant variance), Normality of residuals, No multicollinearity.

### 2. How do you interpret the coefficients?

* A coefficient tells how much the target variable changes with a 1-unit change in that feature, holding all other features constant.

### 3. What is R² score and its significance?

* R² measures how well the model explains variance in the target variable. Ranges from 0 to 1.

### 4. When would you prefer MSE over MAE?

* Use MSE when you want to penalize large errors more heavily. MAE treats all errors equally.

### 5. How do you detect multicollinearity?

* Use correlation matrix or Variance Inflation Factor (VIF). High VIF (>10) indicates multicollinearity.

### 6. What is the difference between simple and multiple regression?

* Simple regression: 1 independent variable. Multiple regression: 2 or more independent variables.

### 7. Can linear regression be used for classification?

* No. It is used for predicting continuous variables. For classification, use logistic regression or classification algorithms.

### 8. What happens if you violate regression assumptions?

* Model predictions may become biased, inefficient, or invalid. It may lead to incorrect interpretations and conclusions.

---

## Dataset

* You can use any relevant dataset (e.g., `Housing.csv`) with features like number of bedrooms, area, furnishing status, etc.
* Download example dataset: *\[click here]*

