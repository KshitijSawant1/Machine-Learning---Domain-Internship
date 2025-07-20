import pandas as pd

# Step 1: Load the CSV file
df = pd.read_csv("Session 1/Part_3/Samplecal.csv")

# Step 2: Display basic information
print("Dataset Loaded Successfully")
print("Shape of the dataset:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nLast 5 Rows:\n", df.tail())
print("\nData Types and Non-Null Counts:")
print(df.info())

# Step 3: Display summary statistics
print("\nDescriptive Statistics:\n", df.describe())

# Step 4: Check for missing values
print("\nMissing Values in Each Column:\n", df.isnull().sum())

# Optional: Visualize missing data using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Value Heatmap")
plt.show()
