```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
```

* This imports the required libraries:

  * `fetch_california_housing`: Loads the **California Housing** dataset from `sklearn.datasets`.
  * `pandas`: Used for handling tabular data via DataFrames.

---

```python
data = fetch_california_housing()
```

* This loads the **California Housing dataset**, which includes various features like:

  * `MedInc` (Median income in a block)
  * `HouseAge` (Median house age)
  * `AveRooms`, `AveBedrms`, `Population`, etc.
* `data` is a **Bunch object** (similar to a dictionary), containing:

  * `data.data`: Feature values.
  * `data.feature_names`: Names of the features.
  * `data.target`: The target variable (median house value).

---

```python
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
```

* Converts the feature matrix into a **pandas DataFrame** with proper column names.
* Adds a new column `'Target'` representing the house prices (in \$100,000s).

---

```python
print("Skewness of Features:")
print(df.skew(numeric_only=True).sort_values(ascending=False))
```

* `df.skew(numeric_only=True)`:

  * Calculates the **skewness** for each numeric column.
  * Skewness measures the **asymmetry** of the distribution:

    * `0`: perfectly symmetrical (normal distribution).
    * `>0`: right-skewed (tail on the right).
    * `<0`: left-skewed (tail on the left).

* `.sort_values(ascending=False)`:

  * Sorts the skewness values in **descending** order so you can see the most skewed features at the top.

---

### Purpose of This Code:

To **understand the distribution** of each feature and identify whether any features are **highly skewed**, which can affect certain ML models and may require **normalization or transformation**.