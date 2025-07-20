import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [5, 7, 4, 6, 8]

# Create scatter plot
plt.scatter(x, y, color='blue', marker='o')

# Add titles and labels
plt.title("Simple Scatter Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# Show the plot
plt.show()
