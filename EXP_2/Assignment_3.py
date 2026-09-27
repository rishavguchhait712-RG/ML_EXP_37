import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Correlation matrix
corr = df.corr()

# Heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Find strongest positive correlation
corr_pairs = corr.where(~pd.np.eye(corr.shape[0], dtype=bool)).stack()
print("Strongest positive correlation:")
print(corr_pairs.idxmax(), "=", corr_pairs.max())