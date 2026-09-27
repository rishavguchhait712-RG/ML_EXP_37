import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([
    [600, 1],
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1400, 3],
    [1600, 4],
    [1800, 4],
    [2000, 5]
])

y = np.array([24, 31, 39, 47, 54, 62, 69, 77])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

new_house = np.array([[1300, 3]])
predicted_price = model.predict(new_house)

print("Predicted House Price:",
      round(predicted_price[0], 2),
      "lakh")

print("\nCoefficient of Area:", round(model.coef_[0], 2))
print("Coefficient of Bedrooms:", round(model.coef_[1], 2))
print("Intercept:", round(model.intercept_, 2))

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n--- Evaluation Metrics ---")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))