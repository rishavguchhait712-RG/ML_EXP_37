import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "Age": [18, 22, 26, 30, 34],
    "Salary": [15000, 25000, 35000, 45000, 55000]
}

df = pd.DataFrame(data)

# Standard Scaling
std_scaler = StandardScaler()
standard_df = pd.DataFrame(
    std_scaler.fit_transform(df),
    columns=df.columns
)

# Min-Max Scaling
mm_scaler = MinMaxScaler()
minmax_df = pd.DataFrame(
    mm_scaler.fit_transform(df),
    columns=df.columns
)

print("--- Original Data ---")
print(df)

print("\n--- StandardScaler Output ---")
print(standard_df.round(3))

print("\n--- MinMaxScaler Output ---")
print(minmax_df)

print("\n--- StandardScaler Range ---")
print("Minimum:", round(standard_df.min().min(), 3))
print("Maximum:", round(standard_df.max().max(), 3))

print("\n--- MinMaxScaler Range ---")
print("Minimum:", minmax_df.min().min())
print("Maximum:", minmax_df.max().max())