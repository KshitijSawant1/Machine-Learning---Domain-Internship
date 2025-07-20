### **Imports**

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

- `fetch_california_housing`: Loads the **California Housing** dataset from scikit-learn.
- `pandas`: Used for creating and managing the dataset as a DataFrame.
- `seaborn`: A visualization library built on top of matplotlib with prettier default plots.
- `matplotlib.pyplot`: Used to display the plot.

---

### **Load the Dataset**

```python
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
```

- `data = fetch_california_housing()`
  Loads the housing dataset containing features like income, house age, number of rooms, etc.

- `pd.DataFrame(data.data, columns=data.feature_names)`
  Converts the feature data into a Pandas DataFrame with appropriate column names.

- `df['Target'] = data.target`
  Adds the **target variable** (i.e., median house value) as a new column in the DataFrame.

---

### **Plot the Distribution of the Target (House Prices)**

```python
sns.histplot(df['Target'], bins=50, kde=True)
```

- `sns.histplot(...)`: Creates a histogram to show the **distribution of values** in the `Target` column.
- `bins=50`: Divides the target values into 50 equal-width intervals.
- `kde=True`: Adds a **Kernel Density Estimate (KDE)** curve to show the smoothed distribution line.

---

### **Add Labels and Show Plot**

```python
plt.title("Distribution of House Prices (Target)")
plt.xlabel("Median House Value ($100,000s)")
plt.ylabel("Frequency")
plt.show()
```

- Adds a **title** and axis **labels** to the plot.
- `plt.show()` displays the histogram with the KDE overlay.

---

Here are **customization options** you can apply to your `sns.histplot()` and `matplotlib` setup to tailor the appearance of the distribution plot:

---

## ✅ **Customization Options for `sns.histplot()`**

### 1. **Color Options**

```python
sns.histplot(df['Target'], color='green')
```

You can use: `'blue'`, `'red'`, `'orange'`, `'purple'`, `'black'`, etc.

---

### 2. **Change Bin Size**

```python
sns.histplot(df['Target'], bins=30)  # Fewer bins
sns.histplot(df['Target'], bins=100) # More bins
```

---

### 3. **Remove KDE**

```python
sns.histplot(df['Target'], bins=50, kde=False)
```

---

### 4. **Add Transparency**

```python
sns.histplot(df['Target'], bins=50, alpha=0.6)
```

---

### 5. **Set Edge Color**

```python
sns.histplot(df['Target'], bins=50, edgecolor='black')
```

---

### 6. **Orientation (Horizontal Histogram)**

```python
sns.histplot(df['Target'], bins=50, orientation='horizontal')
```

---

## ✅ **Matplotlib Customizations**

### 7. **Gridlines**

```python
plt.grid(True)
```

---

### 8. **Change Font Sizes**

```python
plt.title("Distribution", fontsize=16)
plt.xlabel("House Price", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
```

---

### 9. **Change Figure Size**

```python
plt.figure(figsize=(12, 6))
sns.histplot(df['Target'], bins=50, kde=True)
```

---

### 10. **Add Vertical Line for Mean or Median**

```python
import numpy as np
plt.axvline(np.mean(df['Target']), color='red', linestyle='--', label='Mean')
plt.axvline(np.median(df['Target']), color='blue', linestyle=':', label='Median')
plt.legend()
```

---

## Example: Combined Version

```python
plt.figure(figsize=(12, 6))
sns.histplot(df['Target'], bins=50, kde=True, color='skyblue', edgecolor='black', alpha=0.8)
plt.axvline(np.mean(df['Target']), color='red', linestyle='--', label='Mean')
plt.axvline(np.median(df['Target']), color='blue', linestyle=':', label='Median')
plt.title("Distribution of House Prices", fontsize=16)
plt.xlabel("Median House Value ($100,000s)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
```

Let me know if you want to customize for dark mode, add annotation text, or save the plot as an image.
