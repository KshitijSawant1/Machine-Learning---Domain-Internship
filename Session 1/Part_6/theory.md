### Code:

```python
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
```

---

### 🔍 Explanation:

* `x` and `y` are lists representing data points.
* `plt.scatter()` creates the scatter plot.
* `color='blue'` sets the dot color.
* `marker='o'` makes the points circular.
* `plt.title()`, `plt.xlabel()`, and `plt.ylabel()` set the title and axis labels.

---

### 1. **Change Marker Size (`s`)**

```python
plt.scatter(x, y, s=100)  # Larger dots
```

---

### 2. **Change Marker Style (`marker`)**

Common marker options:

* `'o'` – Circle (default)
* `'s'` – Square
* `'^'` – Triangle Up
* `'D'` – Diamond
* `'*'` – Star

```python
plt.scatter(x, y, marker='^')  # Triangle markers
```

---

### 3. **Change Colors (`color`)**

You can specify:

* Named colors: `'red'`, `'green'`, `'purple'`
* Hex codes: `'#FF5733'`
* RGB tuples: `(0.2, 0.4, 0.6)`

```python
plt.scatter(x, y, color='green')
```

---

### 4. **Add Categories with Color Coding**

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y = [5, 7, 4, 6, 8, 3]
categories = ['A', 'B', 'A', 'B', 'A', 'B']
colors = ['red' if cat == 'A' else 'blue' for cat in categories]

plt.scatter(x, y, color=colors)
plt.title("Category-wise Scatter Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()
```

---

### 5. **Add Transparency (`alpha`)**

```python
plt.scatter(x, y, alpha=0.6)  # 0 (transparent) to 1 (opaque)
```

---

### 6. **Edge Color and Line Width**

```python
plt.scatter(x, y, edgecolors='black', linewidths=1.5)
```

---

### ✅ 7. **Color by Value (Colormap)**

```python
import numpy as np

x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)

plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar()  # Show color scale
plt.title("Color-Mapped Scatter Plot")
plt.show()
```
