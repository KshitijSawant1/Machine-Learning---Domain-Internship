import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
df = pd.DataFrame({
    'Score': [88, 92, 80, 89, 100, 76, 85, 90],
    'Group': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
})

# Create box plot
sns.boxplot(x='Group', y='Score', data=df)

# Add title
plt.title("Box Plot by Group")

# Show plot
plt.show()
