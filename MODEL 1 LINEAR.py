import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

# Load Dataset
df = pd.read_csv(r"C:\Users\manthan barot\OneDrive\Desktop\WELTECH\MACHINE LEARNING\ML TEST - 13.6.26\ml_datasets\01_linear_regression_house_price.csv")

# Check Missing Values
print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Fill Missing Values with Mean
df["House_Size_sqft"] = df["House_Size_sqft"].fillna(
    df["House_Size_sqft"].mean()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Features and Target
X = df[["House_Size_sqft"]]
y = df["Price_INR"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Model Creation
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Results")
print("MAE         :", mae)
print("RMSE        :", rmse)
print("R2 Score    :", r2)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5)
plt.plot(X_test, y_pred, color="red")
plt.xlabel("House Size (sqft)")
plt.ylabel("Price (INR)")
plt.title("House Price Prediction using Linear Regression")
plt.show()

# New Prediction
new_house = pd.DataFrame({"House_Size_sqft": [2500]})

predicted_price = model.predict(new_house)

print("\nPredicted Price for 2500 sqft House:")
print(predicted_price[0])