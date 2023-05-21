# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:10:32 2023

@author: shaikjamal
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('D:/Neural Networks/gas_turbines.csv')
print(data.head())
print(data.shape)
print(data.describe())

# Step 2: Explore the data
# Perform EDA, check for missing values, visualize distributions, etc.

# Step 3: Prepare the data
X = data.drop("TEY", axis=1)  # Features (ambient variables)
y = data["TEY"]               # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Select a machine learning algorithm
model = LinearRegression()  # Use linear regression as an example

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)

# Step 7: Tune the model (if necessary)

# Step 8: Make predictions
new_data = pd.DataFrame(...)  # New unseen data
predicted_tey = model.predict(new_data)
