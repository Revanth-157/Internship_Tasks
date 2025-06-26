import matplotlib.pyplot as plt
import pandas as pds
import sklearn as sk

# File path
file = "Housing.csv"

# Task 1: Import and Preprocess the Dataset

# Option A: Using pandas
data = pds.read_csv(file)
for i in data.columns:
    if data[i].dtype in ["float64", "int64"]:
        data[i].fillna(data[i].mean())
    else:
        data[i].fillna(data[i].mode()[0])
data = pds.get_dummies(data, drop_first=True)

# Option B: Using sklearn
data1 = pds.read_csv(file)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Separate numerical and categorical columns
num_cols = data1.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data1.select_dtypes(include=['object']).columns

# Impute missing values
imputer = SimpleImputer(strategy='mean')
imputer1 = SimpleImputer(strategy='most_frequent')
data1[num_cols] = imputer.fit_transform(data1[num_cols])
data1[cat_cols] = imputer1.fit_transform(data1[cat_cols])

# Encode categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cat = encoder.fit_transform(data1[cat_cols])
encoded_cat_df = pds.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

# Merge encoded and numeric data
data1.drop(columns=cat_cols, inplace=True)
data1 = pds.concat([data1, encoded_cat_df], axis=1)

# Normalize numerical features
scaler = StandardScaler()
data1[num_cols] = scaler.fit_transform(data1[num_cols])

# Task 2: Split Data into Train-Test Sets
from sklearn.model_selection import train_test_split

# Replace 'price' with your actual target column name in the dataset
target = 'price'

X = data1.drop(columns=[target])
y = data1[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 3: Fit a Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Task 4: Evaluate the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Task 5: Plot Regression Results and Interpret Coefficients
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Line of perfect fit
plt.show()

# Coefficients and intercept
print("Intercept:", model.intercept_)
print("Feature Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef}")
# This code completes the tasks of importing, preprocessing, splitting, fitting, evaluating, and visualizing a linear regression model on a housing dataset.