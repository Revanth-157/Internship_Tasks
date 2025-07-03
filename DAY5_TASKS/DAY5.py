# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Heart Disease dataset
# Replace with your local file path or use a direct link if available
df = pd.read_csv("heart.csv")  # CSV must be pre-downloaded

# Display first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Separate features and target
X = df.drop("target", axis=1)  # Features
y = df["target"]               # Target label (1 = heart disease, 0 = no disease)

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------
# 1. Train a Decision Tree Classifier
# --------------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# --------------------------------------
# 2. Analyze Overfitting and Control Tree Depth
# --------------------------------------
depths = range(1, 21)
train_scores = []
test_scores = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

# Plot accuracy vs tree depth
plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, label='Train Accuracy')
plt.plot(depths, test_scores, label='Test Accuracy')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Effect of Tree Depth on Overfitting")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------
# 3. Train a Random Forest and Compare Accuracy
# --------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# --------------------------------------
# 4. Interpret Feature Importances
# --------------------------------------
importances = rf.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=features.index)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# --------------------------------------
# 5. Evaluate Using Cross-Validation
# --------------------------------------
cv_scores_dt = cross_val_score(dt, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)

print("\nCross-validation Accuracy (Decision Tree):", cv_scores_dt.mean())
print("Cross-validation Accuracy (Random Forest):", cv_scores_rf.mean())
