from sklearn.datasets import fetch_openml
import pandas as pd

# Step 1: Load the dataset from OpenML
# The Titanic dataset is widely used for classification tasks
data = fetch_openml(name='titanic', version=1, as_frame=True)

# Step 2: Convert it to a Pandas DataFrame
df = data.frame

# Step 3: Preview the dataset
print("Titanic dataset loaded successfully from OpenML!")
print("Shape:", df.shape)
print(df.head())
