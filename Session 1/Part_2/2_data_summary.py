from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

print("Dataset Shape:", df.shape)
print("\nBasic Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Print the first 5 rows
print("\nHead (First 5 rows):")
print(df.head())

# Print the last 5 rows
print("\nTail (Last 5 rows):")
print(df.tail())

# Minimum value in each column
print("\nMinimum Values:")
print(df.min())

# Maximum value in each column
print("\nMaximum Values:")
print(df.max())

# Count of non-null entries per column
print("\nCount of Non-Null Entries:")
print(df.count())

# Median value for each column
print("\nMedian Values:")
print(df.median())

# Variance of each column
print("\nVariance:")
print(df.var())

# Number of unique values in each column (useful for categorical data)
print("\nUnique Value Counts:")
print(df.nunique())
