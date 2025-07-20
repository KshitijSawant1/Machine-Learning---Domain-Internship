
### Code Breakdown

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

* **`seaborn`**: For advanced statistical plotting (built on top of matplotlib).
* **`matplotlib.pyplot`**: Basic plotting library.
* **`pandas`**: For creating and handling tabular data (like DataFrames).

---

```python
df = pd.DataFrame({
    'Score': [88, 92, 80, 89, 100, 76, 85, 90],
    'Group': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
})
```

* A **DataFrame `df`** is created with two columns:

  * `'Score'`: numerical values representing test scores.
  * `'Group'`: categorical labels (`'A'` or `'B'`), representing group membership.

---

```python
sns.boxplot(x='Group', y='Score', data=df)
```

* This creates a **box plot** where:

  * **`x='Group'`**: Groups A and B are shown on the x-axis.
  * **`y='Score'`**: The corresponding scores are plotted on the y-axis.
  * The plot shows:

    * Median (central line in box)
    * Interquartile range (box edges: 25th and 75th percentiles)
    * Whiskers (range, excluding outliers)
    * Any outliers (individual points beyond whiskers)

---

```python
plt.title("Box Plot by Group")
```

* Sets the title of the plot.

---

```python
plt.show()
```

* Displays the plot window.

---

### 📊 What You Learn from This Plot

* Compare the distribution of scores in **Group A vs. Group B**.
* Check for differences in **spread, median, and outliers** between groups.

