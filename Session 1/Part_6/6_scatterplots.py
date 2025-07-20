from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

sns.scatterplot(x='MedInc', y='Target', data=df, alpha=0.3)
plt.title("Median Income vs House Value")
plt.show()

sns.scatterplot(x='AveRooms', y='Target', data=df, alpha=0.3)
plt.title("Average Rooms vs House Value")
plt.show()
