import sklearn as sk
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pds
import numpy as np
#Objective:-Using SVMs for linear as well as non-linear
#Tools: Scikit-learn, Pandas, NumPy
#Task 1:- Load and prepare datset for binary classification
# Importing the breast cancer dataset
file="breast-cancer.csv"
data=pds.read_csv(file)
#Removing the 'id' column if it exists
if 'id' in data.columns:
    data.drop(columns=["id"],inplace=True)
#Conversion of categorical numerical values into binary values(Yes/No or 1/0) for binary classification for SVM
for column in data.columns:
    if data[column].dtype == 'object':
        data[column]=data[column].map({"M":1,"B":0})    #M for malignant and B for benign
    else:
        data[column]=sk.preprocessing.MinMaxScaler().fit_transform(data[column].values.reshape(-1,1))
        data[column]=(data[column]>0.2).astype(int)
#Dividing data into trainning and testing data
X=data.drop(columns=["diagnosis"])
Y=data["diagnosis"]
X_train,X_test,y_train,y_test=sk.model_selection.train_test_split(X,Y,test_size=0.2,random_state=42)
scaler=sk.preprocessing.StandardScaler()
X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.fit_transform(X_test)
#Task-2:- Train SVM with linear and RBF kernel
#Training linear model
model_linear=SVC(kernel='linear',C=1,probability=True)  # Using sigmoid kernel
model_linear.fit(X_train_scale,y_train)
#Prediction data
predict=model_linear.predict(X_test_scale)
data_accuracy=model_linear.score(X_test_scale,y_test)
print("Accuracy",data_accuracy)
#Training RBF(Radial Basis Function)
model_RBF=SVC(kernel="rbf",probability=True)
model_RBF.fit(X_train_scale,y_train)
predict=model_RBF.predict(X_test_scale)
data_accuracy=model_RBF.score(X_test_scale,y_test)
print("Accuracy",data_accuracy)
#Task-3:- Visualize the decision boundary using 2D data
#Visualization of decision boundary
#Reducing the data to 2D 
# Select 2 features (Radius Mean and Texture Mean)
X = data[["radius_mean", "texture_mean"]].values
y = data["diagnosis"].values  # Binary labels

# Train/test split
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
 # Taking only the first two features for visualization
scaler=sk.preprocessing.StandardScaler()
X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.fit_transform(X_test)
#Training linear model 
model_linear=SVC(kernel='linear',C=1,probability=True)  # Using sigmoid kernel
model_linear.fit(X_train_scale,y_train)
#Visualization of decision boundary for linear model
#Creating a meshgrid for visualization  
x_min,x_max=X_train_scale[:,0].min()-1,X_train_scale[:,0].max()+1
y_min,y_max=X_train_scale[:,1].min()-1,X_train_scale[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
#Predicting the class labels for each point in the meshgrid
Z=model_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z=Z.reshape(xx.shape)
#Plotting the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
#Plotting the training points
plt.scatter(X_train_scale[:,0],X_train_scale[:,1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.title('Decision Boundary for Linear SVM')
plt.show()
#FOr RBF Model decision boundary 
model_linear=SVC(kernel='rbf',C=1,probability=True)  # Using sigmoid kernel
model_linear.fit(X_train_scale,y_train)
#Visualization of decision boundary for linear model
#Creating a meshgrid for visualization  
x_min,x_max=X_train_scale[:,0].min()-1,X_train_scale[:,0].max()+1
y_min,y_max=X_train_scale[:,1].min()-1,X_train_scale[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
#Predicting the class labels for each point in the meshgrid
Z=model_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z=Z.reshape(xx.shape)
#Plotting the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
#Plotting the training points
plt.scatter(X_train_scale[:,0],X_train_scale[:,1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.title('Decision Boundary for Linear SVM')
plt.show()

# Task 4: Hyperparameter Tuning for RBF Kernel SVM
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],   # Kernel coefficient
    'kernel': ['rbf']                 # Only RBF kernel here
}

# Instantiate GridSearchCV with 5-fold cross-validation
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, scoring='accuracy')
grid.fit(X_train_scale, y_train)

# Best parameters and score
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

# Final evaluation on test set
best_model = grid.best_estimator_
test_accuracy = best_model.score(X_test_scale, y_test)
print("Test Accuracy with Best Parameters:", test_accuracy)




from sklearn.model_selection import cross_val_score

# Compare SVM with different kernels using 5-fold cross-validation
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    model = SVC(kernel=kernel, C=1)
    scores = cross_val_score(model, X_train_scale, y_train, cv=5, scoring='accuracy')
    print(f"\nKernel: {kernel}")
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", scores.mean())


    
