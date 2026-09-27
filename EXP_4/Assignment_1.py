# Practice Problem 1 - House Price Prediction

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dataset
area = [600, 800, 1000, 1200, 1400, 1600]
price = [25, 32, 40, 47, 55, 63]

X = np.array(area).reshape(-1, 1)
y = np.array(price)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Predict price for a new house
area_new = [[1100]]
price_new = model.predict(area_new)

print("Predicted price for 1100 sq. ft.:", price_new[0])

# Calculate evaluation metrics
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print("\n--- Evaluation Metrics ---")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

# Plot
plt.scatter(X, y, label="Actual Prices")
plt.plot(X, predictions, label="Regression Line")

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price (Lakh)")
plt.title