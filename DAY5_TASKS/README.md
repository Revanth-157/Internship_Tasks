# Heart Disease Prediction using Decision Trees and Random Forest

This project demonstrates the use of **Decision Tree** and **Random Forest** classifiers for predicting heart disease using the UCI Heart Disease dataset.

---

## 🔍 Dataset

- Source: [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- Target: `target` column (1 = Disease, 0 = No Disease)
- Features: Age, sex, cp (chest pain), trestbps, chol, thalach, etc.

---

## 🧠 Mini Guide (Hints)

1. **Train a Decision Tree Classifier and visualize the tree**  
   - Used `DecisionTreeClassifier` and `plot_tree` from sklearn.

2. **Analyze overfitting and control tree depth**  
   - Varied `max_depth` from 1–20 and plotted accuracy to show overfitting.

3. **Train a Random Forest and compare accuracy**  
   - Used `RandomForestClassifier` with 100 trees.

4. **Interpret feature importances**  
   - Used `.feature_importances_` and visualized with Seaborn barplot.

5. **Evaluate using cross-validation**  
   - Used `cross_val_score` with `cv=5`.

---

## 🧑‍💼 Interview Questions & Answers

1. **How does a decision tree work?**  
   - It splits data based on feature values using metrics like entropy or Gini to create a tree that predicts the target.

2. **What is entropy and information gain?**  
   - Entropy measures impurity; Information Gain is the reduction in entropy after a split.

3. **How is random forest better than a single tree?**  
   - It reduces overfitting and variance by averaging multiple decision trees (ensemble learning).

4. **What is overfitting and how do you prevent it?**  
   - Overfitting means model memorizes the data instead of generalizing. Prevent by pruning, limiting depth, or using ensemble methods.

5. **What is bagging?**  
   - Bagging (Bootstrap Aggregating) trains multiple models on random subsets and aggregates results.

6. **How do you visualize a decision tree?**  
   - Using `plot_tree()` from `sklearn.tree`.

7. **How do you interpret feature importance?**  
   - Higher importance means the feature had greater impact on decisions (based on Gini or entropy reductions).

8. **What are the pros/cons of random forests?**  
   - ✅ High accuracy, robust to overfitting  
     ❌ Less interpretable, slower for large datasets

---

## ✅ Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
