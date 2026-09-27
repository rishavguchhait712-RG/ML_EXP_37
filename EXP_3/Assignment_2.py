import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "Age": [18, 22, 26, 30, 34],
    "Salary": [15000, 25000, 35000, 45000, 55000]
}

df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

print("\n--- Data after MinMax Scaling ---")
print(scaled_df)