import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 14, 11]

# Create line plot
plt.plot(x, y, marker='o', linestyle='-', color='blue')

# Add labels and title
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Simple Line Graph")

# Show the graph
plt.grid(True)
plt.show()
