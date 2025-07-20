from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['MedInc', 'AveRooms', 'AveOccup']])
plt.title("Boxplot for Selected Features")
plt.show()
