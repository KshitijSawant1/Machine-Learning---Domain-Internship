from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

print("Skewness of Features:")
print(df.skew(numeric_only=True).sort_values(ascending=False))
