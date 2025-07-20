from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

sns.pairplot(df.sample(300), vars=['MedInc', 'HouseAge', 'AveRooms', 'Target'])
