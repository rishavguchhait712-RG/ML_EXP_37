import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([[600,1], [800,2], [1000,2],
              [1200,3], [1400,3], [1600,4]])

y = np.array([24, 32, 39, 47, 55, 63])

model = LinearRegression()
model.fit(X, y)

# Prediction
new_house = [[1300, 3]]
prediction = model.predict(new_house)

print("Predicted Price:", round(prediction[0], 2), "lakh")

# Evaluation
y_pred = model.predict(X)

print("MAE:", round(mean_absolute_error(y, y_pred), 2))
print("MSE:", round(mean_squared_error(y, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y, y_pred)), 2))
print("R2:", round(r2_score(y, y_pred), 4))