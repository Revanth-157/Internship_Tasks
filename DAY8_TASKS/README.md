# K-Means Clustering: Mall Customer Segmentation

## 🧠 Objective
Apply K-Means clustering to segment mall customers based on their spending behavior and income using unsupervised learning.

---

## 📊 Dataset
**Mall_Customers.csv**  
Columns used:
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

## 🛠️ Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🗂️ Mini Guide

| Task | Description |
|------|-------------|
| 1 | Load and visualize the dataset (optional PCA for 2D view if needed) |
| 2 | Fit K-Means and assign cluster labels |
| 3 | Use the Elbow Method to find optimal K |
| 4 | Visualize clusters with color-coding |
| 5 | Evaluate clustering using Silhouette Score |

---

## 📌 Interview Questions

### 1. How does K-Means clustering work?
K-Means partitions data into `k` clusters by minimizing the variance within each cluster using centroids and iterative reassignment.

### 2. What is the Elbow method?
It's a method to find the optimal number of clusters by plotting inertia vs `k` and identifying the point where the decrease slows down (elbow point).

### 3. What are the limitations of K-Means?
- Assumes spherical clusters
- Sensitive to outliers
- Requires pre-defined `k`
- May converge to local minima

### 4. How does initialization affect results?
Poor centroid initialization may lead to suboptimal clusters. Using `k-means++` helps improve initialization.

### 5. What is inertia in K-Means?
It’s the sum of squared distances between data points and their closest cluster center. Lower inertia means tighter clusters.

### 6. What is Silhouette Score?
It measures how similar a point is to its own cluster vs other clusters. Ranges from -1 to 1 (higher is better).

### 7. How do you choose the right number of clusters?
Use metrics like Elbow Method, Silhouette Score, or domain knowledge.

### 8. What’s the difference between clustering and classification?
Clustering is unsupervised (no labels); classification is supervised (uses labeled data).

---

## ✅ How to Run

1. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn
