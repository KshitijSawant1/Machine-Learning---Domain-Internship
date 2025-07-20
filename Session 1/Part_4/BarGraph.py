import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

# Create bar chart
plt.bar(categories, values, color='skyblue')

# Add titles and labels
plt.title("Simple Bar Graph")
plt.xlabel("Categories")
plt.ylabel("Values")

# Show the graph
plt.show()
