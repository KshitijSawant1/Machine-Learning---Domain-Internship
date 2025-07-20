## Fixing SSL Certificate Error for `fetch_openml()` in scikit-learn on macOS

### Problem:

When using `fetch_openml()` from `sklearn.datasets`, you encountered this error:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

This is a common issue on macOS because **Python installed from python.org doesn’t automatically trust the macOS keychain certificates**. As a result, HTTPS connections to OpenML (or any secure site) fail.

---

## Solution: Install Missing Certificates for Python

### Step 1: Identify Python Version

Run:

```bash
python3 --version
```

Example Output:

```
Python 3.11.x
```

This confirms your Python version is installed from the official installer (not via Homebrew or conda).

---

### Step 2: Locate the `Install Certificates.command` Script

Navigate to the folder where your Python is installed. Typically, it’s:

```bash
/Applications/Python\ 3.11/
```

You can verify:

```bash
ls /Applications | grep Python
```

---

###  Step 3: Run the Installer Script

Run this command in the terminal:

```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

💡 This script automatically installs `certifi` and links macOS certificates so that your Python can securely connect via HTTPS.

---

### Step 4: Retry Your OpenML Code

Now the following code works without SSL errors:

```python
from sklearn.datasets import fetch_openml
import pandas as pd

data = fetch_openml(name='titanic', version=1, as_frame=True)
df = data.frame

print("Titanic dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())
```

---
