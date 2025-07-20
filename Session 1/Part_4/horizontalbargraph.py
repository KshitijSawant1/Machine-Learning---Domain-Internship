import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

# Horizontal bar chart
plt.barh(categories, values, color='lightcoral')

# Labels and title
plt.title("Horizontal Bar Chart")
plt.xlabel("Values")
plt.ylabel("Categories")
plt.tight_layout()
plt.show()
