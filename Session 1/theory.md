# Step 1: Python Environment Setup for Machine Learning

## Option 1: Local Setup (Recommended for full control)

### Requirements:

* Python 3.8+ (Use [Python.org](https://www.python.org/downloads/))
* pip (comes pre-installed with Python)
* Virtual Environment (optional but clean)

---

### A. Install Python (if not installed)

Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)

Check version:

```bash
python --version
# or
python3 --version
```

---

### B. (Optional) Create a Virtual Environment

```bash
# Create env
python -m venv ml_env

# Activate env
# Windows
ml_env\Scripts\activate
# macOS/Linux
source ml_env/bin/activate
```

---

### C. Install Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Optional: For deep learning (used in Session 4)

```bash
pip install tensorflow keras
```

---

### D. Start Jupyter Notebook

```bash
jupyter notebook
```

Or install and use JupyterLab (better UI):

```bash
pip install jupyterlab
jupyter lab
```

---

## Option 2: Google Colab (No installation required)

> Visit [https://colab.research.google.com](https://colab.research.google.com)

* Sign in with Google
* Create a new notebook
* Start coding immediately
* Use `!pip install` for any extra libraries:

```python
!pip install seaborn scikit-learn
```

---

## Common Package Imports (Put this at the top of your notebooks)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

---

## Summary

| Tool         | For What                | Command/Link                                                   |
| ------------ | ----------------------- | -------------------------------------------------------------- |
| Python       | Core language           | `python --version`                                             |
| pip          | Install libraries       | `pip install <libname>`                                        |
| venv         | Isolated environments   | `python -m venv env_name`                                      |
| Jupyter      | Code notebook interface | `jupyter notebook` or `jupyter lab`                            |
| Google Colab | Cloud-based notebook    | [colab.research.google.com](https://colab.research.google.com) |

---

Would you like:

* A **requirements.txt** file to easily share or recreate the environment?
* A ready-to-use **Colab link** with all dependencies set?

Let me know where you'd like to code — local Jupyter or Colab — and I’ll generate the base EDA notebook or script accordingly.
