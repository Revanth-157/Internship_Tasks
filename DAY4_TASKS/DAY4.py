import sklearn as sk
import pandas as pds
import numpy as np
#Tasks
#1.Choose a binary classification dataset.
file='data.csv'
pds.set_option('display.max_columns', None)  # Show all columns in the DataFrame
pds.set_option('display.max_rows',None)  # Set all rows
data=pds.read_csv(file)
data['diagnosis']=data['diagnosis'].map({"M":1,"B":0})
for i in data.columns:
    if i!="diagnosis" and i!="id" and data[i].dtype in ["float64", "int64"]:
        data[i]=sk.preprocessing.MinMaxScaler().fit_transform(data[[i]])
        data[i]=(data[i]>0.2).astype(int)  # Binarize the feature
#2.Train/test split and standardize features.
# Drop the 'id' column (not useful for prediction)
data = data.drop(columns=["id"])

#Split data into features (X) and target (y)
X = data.drop(columns=["diagnosis"])  # All columns except target
y = data["diagnosis"]  
# Train-test split
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

#3.Fit a Logistic Regression model.
model=sk.linear_model.LogisticRegression()
model.fit(X_train, y_train)
#4.Evaluate with confusion matrix, precision, recall, ROC-AUC.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probabilities for the positive class
#Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))  
print("ROC-AUC:", roc_auc_score(y_test, y_proba))       
#5.Plot ROC curve and precision-recall curve.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve 
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend()
plt.show()
#5.Tune threshold and explain sigmoid function.
# Custom threshold
threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)

print("Custom Threshold Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
print("Precision (custom):", precision_score(y_test, y_pred_custom))
print("Recall (custom):", recall_score(y_test, y_pred_custom))
    
