import matplotlib.pyplot as plt
import numpy as np

# Sample data
labels = ['Q1', 'Q2', 'Q3', 'Q4']
group1 = [20, 34, 30, 35]
group2 = [25, 32, 34, 20]

x = np.arange(len(labels))  # [0, 1, 2, 3]
width = 0.35

# Grouped bars
plt.bar(x - width/2, group1, width, label='Product A', color='steelblue')
plt.bar(x + width/2, group2, width, label='Product B', color='orange')

# Labels and title
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.title('Quarterly Sales Comparison')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.show()
