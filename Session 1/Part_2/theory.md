## Code Explanation

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
```

* **`fetch_california_housing`**: A function from `sklearn.datasets` that loads the **California housing dataset**, which includes features like average income, house age, etc., and the target is the median house value.
* **`pandas as pd`**: Imports the pandas library for data manipulation and analysis.

---

```python
data = fetch_california_housing()
```

* Fetches the dataset and stores it in a dictionary-like object called `data`.

---

```python
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
```

* Converts the data matrix into a pandas DataFrame.
* Assigns column names based on `data.feature_names`.
* Adds a new column `'Target'` which contains the **median house value** for each entry.

---

```python
print("Dataset Shape:", df.shape)
```

* Prints the **dimensions** of the dataset (rows × columns).

---

```python
print("\nBasic Info:")
print(df.info())
```

* Displays basic metadata about the DataFrame: column names, data types, non-null counts, memory usage.

---

```python
print("\nStatistical Summary:")
print(df.describe())
```

* Outputs **descriptive statistics** (mean, std, min, 25%, 50%, 75%, max) for all numerical columns.


## Summary of What This Does

| Operation       | Purpose                                     |
| --------------- | ------------------------------------------- |
| `df.head()`     | View the first 5 rows                       |
| `df.tail()`     | View the last 5 rows                        |
| `df.min()`      | Minimum value for each column               |
| `df.max()`      | Maximum value for each column               |
| `df.count()`    | Number of non-null entries per column       |
| `df.describe()` | Summary stats like mean, std, etc.          |
| `df.median()`   | Median of each column                       |
| `df.var()`      | Variance (spread) of values in each column  |
| `df.nunique()`  | Count of unique values (useful in features) |
