# DAY8.py

# Task 1: Load and visualize dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Display first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Extract relevant features: Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Optional: Apply PCA for 2D visualization (not necessary here since we already have 2 features)
# Uncomment below if your dataset has more than 2 dimensions
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# Task 2: Fit K-Means and assign cluster labels (trying with 5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to original data
data['Cluster'] = clusters

# Task 3: Use the Elbow Method to find optimal K
inertias = []
k_values = range(1, 11)
for k in k_values:
    kmeans_k = KMeans(n_clusters=k, random_state=42)
    kmeans_k.fit(X)
    inertias.append(kmeans_k.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Task 4: Visualize clusters with color-coding
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segments (K=5)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# Task 5: Evaluate clustering using Silhouette Score
score = silhouette_score(X, clusters)
print(f'Silhouette Score for K=5: {score:.2f}')
