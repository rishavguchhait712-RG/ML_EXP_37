import pandas as pd
import numpy as np

# Create synthetic dataset
data = {
    "Age": [22, 25, 30, 28, 35, 40, 26, 32],
    "Salary": [25000, 30000, 45000, 40000, 55000, 70000, 35000, 50000],
    "Department": ["IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "IT"],
    "Years_of_Experience": [1, 2, 5, 4, 8, 12, 3, 7]
}

df = pd.DataFrame(data)

# Insert missing values deliberately
df.loc[2, "Age"] = np.nan
df.loc[4, "Salary"] = np.nan
df.loc[6, "Department"] = np.nan
df.loc[7, "Years_of_Experience"] = np.nan

print("Dataset with missing values:")
print(df)

# Preprocess missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])
df["Years_of_Experience"] = df["Years_of_Experience"].fillna(
    df["Years_of_Experience"].mean()
)

print("\nDataset after preprocessing:")
print(df)