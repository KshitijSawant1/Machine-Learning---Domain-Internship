Load Data Code 
---

---
```python
from sklearn.datasets import fetch_california_housing
```

✅ **What it does:**
Imports a function from `scikit-learn` that downloads and loads the **California Housing dataset**.

> ℹ️ This dataset contains information about California districts (median income, house age, rooms, etc.) and the **median house price** — useful for regression tasks.

---

```python
import pandas as pd
```

 **What it does:**
Imports the `pandas` library, which is used for working with tabular data (like spreadsheets and CSVs).

---

```python
data = fetch_california_housing()
```

 **What it does:**
Calls the `fetch_california_housing()` function, which returns a special **Bunch object** containing:

* `data`: the features (inputs like income, rooms, etc.)
* `target`: the output (median house value)
* `feature_names`: names of each input column

Think of it like a dictionary with keys: `'data'`, `'target'`, `'feature_names'`, etc.

---

```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```

 **What it does:**
Creates a **pandas DataFrame** (`df`) using the input features from the dataset. Each column in the table is named according to `data.feature_names`.

> Now you have a structured table of the housing features!

Example columns: `MedInc`, `HouseAge`, `AveRooms`, `Latitude`, etc.

---

```python
df['Target'] = data.target
```

 **What it does:**
Adds a new column called `"Target"` to the DataFrame, which contains the **house price** (in \$100,000s) for each row.

So each row now looks like:

```
[MedInc, HouseAge, AveRooms, ..., Longitude, Target]
```

---

```python
print("Data loaded successfully!")
```

 **What it does:**
Prints a confirmation message so you know the script is working.

---

```python
print(df.head())
```

**What it does:**
Displays the **first 5 rows** of the DataFrame so you can preview the data.

---

## Example Output Looks Like:

| MedInc | HouseAge | AveRooms | ... | Longitude | Target |
| ------ | -------- | -------- | --- | --------- | ------ |
| 8.32   | 41.0     | 6.98     | ... | -122.23   | 4.526  |
| 8.30   | 21.0     | 6.23     | ... | -122.22   | 3.585  |

---

## Final Outcome

At the end of this code:

* `df` is a **clean, labeled table** with both input features and target output
* It’s ready for **EDA (exploration), modeling, and visualization**

---

Load Data Code Open ML
---

---

## Line-by-Line Explanation

```python
from sklearn.datasets import fetch_openml
```

This line imports the `fetch_openml` function from the `sklearn.datasets` module.
This function allows you to load datasets directly from [OpenML](https://www.openml.org), a public platform that hosts hundreds of machine learning datasets.

---

```python
import pandas as pd
```

This imports the `pandas` library, which is widely used for handling and analyzing tabular data. It provides the `DataFrame` structure, which is ideal for working with datasets.

---

```python
# Step 1: Load the dataset from OpenML
# The Titanic dataset is widely used for classification tasks
data = fetch_openml(name='titanic', version=1, as_frame=True)
```

This line fetches the **Titanic dataset** from OpenML:

* `name='titanic'`: Specifies which dataset to load by name.
* `version=1`: Specifies the version of the dataset (version 1 in this case).
* `as_frame=True`: Tells `scikit-learn` to return the data as a pandas `DataFrame` instead of NumPy arrays.

The result is stored in the variable `data`, which is a dictionary-like object (a `Bunch`) containing:

* `data.data`: the feature data
* `data.target`: the target column (if defined)
* `data.feature_names`: the names of the input features
* `data.frame`: a full pandas DataFrame combining both features and target

---

```python
# Step 2: Convert it to a Pandas DataFrame
df = data.frame
```

This line extracts the complete dataset as a `DataFrame` from the fetched OpenML object and assigns it to the variable `df`.

The `df` variable now holds a tabular dataset with rows and columns, ready for analysis.

---

```python
# Step 3: Preview the dataset
print("Titanic dataset loaded successfully from OpenML!")
```

This line prints a confirmation message to indicate that the dataset has been loaded successfully.

---

```python
print("Shape:", df.shape)
```

This line prints the **dimensions** of the dataset, i.e., the number of rows and columns.
The output will look something like:

```
Shape: (1309, 14)
```

Meaning the dataset has 1309 rows and 14 columns.

---

```python
print(df.head())
```

This prints the **first 5 rows** of the dataset using `pandas.DataFrame.head()` to give you a quick preview of the data, including feature names and some example values.

---

## Final Outcome

By the end of this script:

* The Titanic dataset is downloaded from OpenML.
* The full dataset is stored in a clean `pandas` DataFrame.
* The script outputs the shape and the first few records to verify successful loading.

Let me know if you'd like to explore this dataset further with data cleaning or exploratory analysis.

---