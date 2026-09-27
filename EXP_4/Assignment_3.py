# Practice Problem 3 - Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([3, 6, 11, 18, 27, 38])

# Linear Regression
linear = LinearRegression()
linear.fit(X, y)
linear_pred = linear.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X2 = poly.fit_transform(X)

model = LinearRegression()
model.fit(X2, y)
poly_pred = model.predict(X2)

print("Linear Regression R2:", round(r2_score(y, linear_pred), 4))
print("Polynomial Regression R2:", round(r2_score(y, poly_pred), 4))

# Plot
plt.scatter(X, y, label="Actual Data")
plt.plot(X, linear_pred, label="Linear")
plt.plot(X, poly_pred, label="Polynomial")

plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid()
plt.show()