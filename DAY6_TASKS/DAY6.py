# knn_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# ------------------------------------------
# 1. Load Iris.csv Dataset and Normalize Features
# ------------------------------------------
df = pd.read_csv('Iris.csv')

# Drop ID column if exists
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Encode the target class 'species' as integers
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split features and target
X = df.drop('species', axis=1).values
y = df['species'].values
class_names = le.classes_

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------
# 2. Split Data and Train KNN Classifier (K=3)
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ------------------------------------------
# 3. Experiment with Different Values of K
# ------------------------------------------
print("Accuracy for different values of K:")
for k in range(1, 11):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    y_pred_k = knn_k.predict(X_test)
    acc = accuracy_score(y_test, y_pred_k)
    print(f"K={k}: Accuracy = {acc:.2f}")

# ------------------------------------------
# 4. Evaluate Model using Accuracy & Confusion Matrix
# ------------------------------------------
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy (K=3):", acc)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 5. Visualize Decision Boundaries using PCA
# ------------------------------------------
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_train_r, y_train_r)

# Create mesh grid for contour plot
h = 0.02
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
plt.title("KNN Decision Boundary (PCA Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

