import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Plot boxplots
df.boxplot(figsize=(12, 6))

plt.title("Boxplots of Wine Dataset")
plt.xticks(rotation=90)
plt.show()