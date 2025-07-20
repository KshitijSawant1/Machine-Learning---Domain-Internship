from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

sns.histplot(df['Target'], bins=50, kde=True)
plt.title("Distribution of House Prices (Target)")
plt.xlabel("Median House Value ($100,000s)")
plt.ylabel("Frequency")
plt.show()
