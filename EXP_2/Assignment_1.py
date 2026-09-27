import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# Display data
print("First 5 rows:")
print(df.head())

# Basic information
print("\nShape:")
print(df.shape)

print("\nSummary:")
print(df.describe())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Class distribution
print("\nClass distribution:")
print(df["target"].value_counts())

# Histogram
df.hist(figsize=(12, 10))
plt.show()