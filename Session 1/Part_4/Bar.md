

### Code Explanation:

```python
import matplotlib.pyplot as plt
```

* This imports the `matplotlib.pyplot` module as `plt`, which is a common convention.
* `pyplot` provides functions for creating plots, charts, and graphs.

---

```python
# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]
```

* `categories`: These are the labels for each bar on the x-axis (e.g., categories A, B, C, D).
* `values`: These are the heights of the bars corresponding to each category.

---

```python
plt.bar(categories, values, color='skyblue')
```

* This creates a **vertical bar chart**.
* `categories`: provides labels on the x-axis.
* `values`: determines the height of each bar.
* `color='skyblue'`: sets the color of the bars.

---

```python
plt.title("Simple Bar Graph")
plt.xlabel("Categories")
plt.ylabel("Values")
```

* `plt.title`: Adds a title to the graph.
* `plt.xlabel`: Labels the x-axis as “Categories”.
* `plt.ylabel`: Labels the y-axis as “Values”.

---

```python
plt.show()
```

* Displays the graph window.
* It must be called at the end to render the chart.

---

### 🔄 Variations You Can Try:

* `plt.barh(categories, values)` → Horizontal bar chart
* `color='green'` or `color=['red', 'blue', 'orange', 'purple']` → Custom bar colors
* `plt.bar(categories, values, width=0.5)` → Adjust bar width


---

### Syntax Reminder:

```python
plt.bar(x, height, **kwargs)
```

---

### Common Customization Options for `plt.bar()`:

| Option      | Description                        | Example                         |              |
| ----------- | ---------------------------------- | ------------------------------- | ------------ |
| `color`     | Sets bar color (single or list)    | `'skyblue'`, `['red', 'green']` |              |
| `width`     | Sets bar width                     | `width=0.6`                     |              |
| `align`     | Align bars: `'center'` or `'edge'` | `align='edge'`                  |              |
| `edgecolor` | Color of bar borders               | `edgecolor='black'`             |              |
| `linewidth` | Thickness of bar borders           | `linewidth=2`                   |              |
| `hatch`     | Adds pattern (e.g. \`/, \\, -,     | , +, x, o, O, ., \*\`)          | `hatch='//'` |
| `label`     | Label for legend                   | `label='Group A'`               |              |
| `zorder`    | Layering (for overlapping plots)   | `zorder=2`                      |              |

---

###Examples:

#### 1. Bar with border and width

```python
plt.bar(categories, values, color='lightgreen', edgecolor='black', width=0.5)
```

#### 2. Bar with different colors

```python
plt.bar(categories, values, color=['red', 'blue', 'orange', 'purple'])
```

#### 3. Bar with hatch pattern

```python
plt.bar(categories, values, color='skyblue', hatch='//')
```

#### 4. Bar aligned to edge

```python
plt.bar(categories, values, align='edge', width=0.3)
```

#### 5. Bar with label for legend

```python
plt.bar(categories, values, color='steelblue', label='Sales Q1')
plt.legend()
```
