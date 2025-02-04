import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# 1) DATA PREPROCESSING

# i) Load the raw data into pandas dataframe
os.chdir(r"C:\Users\suean\END_TO_END_DATA_SCIENCE_PROJECTS\house-price-prediction")
data = pd.read_csv("data/raw/house_prices_100.csv")
print(data.head())

# ii) Exploratory data analysis (EDA) to understand dataset
# Check for missing values
print(data.isnull().sum())
# Get dataset summary
print(data.info())
print(data.describe())

# iii) Handle Missing Values
# Drop rows or columns with missing values.
# Fill missing values with mean, median, or another strategy.
# Example: Filling missing values with median of column 'bedrooms' (does nothing if no missing values)
data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].median())

# iv) Feature Engineering
# Create new features if necessary.
# Normalize or scale features if needed.

# Example: Scale area to thousands
data['area_in_thousands'] = data['area'] / 1000

# # Example: Min-Max Scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data['area_minmax'] = scaler.fit_transform(data[['area']])

# # Example: Standard Scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data['area_standard'] = scaler.fit_transform(data[['area']])


# v) Split the Data into features (X) and target (y):
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# vi) Split the dataset into training and testing sets. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# vii) Save Processed Data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# Preprocessing can be complex and time consuming, so we usually save it in files instead
# of going straight to building and training the model 

#----------------------------------------------------------
# 2) BUILDING AND TRAINING THE MODEL

# i) Load preprocessed training and testing datasets
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")
# When reading data from a CSV file, pandas.read_csv() returns a DataFrame by default, which is 
# inherently 2D, even if it contains just a single column (becomes (N,1) shaped).

# Convert target variables from 2D arrays ((N, 1) shape) to 1D arrays ((N, 0) shape)
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Many Scikit-learn methods, such as model.fit() and model.predict(), expect target variables (y)
# to be 1-dimensional arrays, not 2D. If y remains as a DataFrame with shape (N, 1), it may raise 
# a warning or error.

# ii) Initialize model (e.g. linear regression)
model = LinearRegression()

# iii) Train the model on the training dataset
model.fit(X_train, y_train)

# iv) Evaluate model
from sklearn.metrics import mean_squared_error, r2_score
# Predict on the test set
y_pred = model.predict(X_test)
# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2 Score):", r2)

# Save the Trained Model
joblib.dump(model, "models/linear_regression_model.pkl")

#--------------------------------------------------------
# 3) EVALUATING THE MODEL ON UNSEEN (TEST) DATA

# i) Load trained model
model = joblib.load("models/linear_regression_model.pkl")

# ii) Load preprocessed test data
X_test = pd.read_csv("data/processed/X_test.csv")
Y_test = pd.read_csv("data/processed/Y_test.csv")

# iii) Use the trained model to predict on the test data
y_pred = model.predict(X_test)

# iv) Evaluate prediction on test data

#a) mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

#b) mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

#c) root mean squared error
rmse = mse ** 0.5
print("Root Mean Squared Error (RMSE):", rmse)

#d) R squared and adjusted R squared
r2 = r2_score(y_test, y_pred)
print("R-squared (R2):", r2)

# Calculate Adjusted R2
n = X_test.shape[0]  # Number of observations
k = X_test.shape[1]  # Number of features
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print("Adjusted R-squared:", adjusted_r2)

# MAE, MSE, RMSE: Lower values indicate better performance.
# 𝑅 squared: A higher value (close to 1) indicates that model explains a 
# large proportion of the variance in the data.
# Adjusted R squared: Use this to verify R squared, especially if have multiple features.

plt.figure(1)
plt.scatter(y_test, y_pred)
# Plot the 45-degree diagonal line (y = x)
min_val = min(min(y_test), min(y_pred))  # Find the minimum value
max_val = max(max(y_test), max(y_pred))  # Find the maximum value
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit (y = x)')
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Predicted vs actual prices')

residuals = y_pred - y_test
plt.figure(2)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted prices')
plt.ylabel('Residuals')
plt.title('Residuals plot')
plt.show()

