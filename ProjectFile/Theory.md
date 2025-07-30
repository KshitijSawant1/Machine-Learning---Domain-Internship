## 📘 Naive Bayes Classifier

Naive Bayes is a **probabilistic classification algorithm** based on **Bayes' Theorem**. It assumes that features are **independent** given the class label — a "naive" assumption, hence the name.

### 🔢 Bayes' Theorem:

$$
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
$$

Where:

* $P(C|X)$: Posterior probability of class C given predictor X
* $P(X|C)$: Likelihood of predictor X given class C
* $P(C)$: Prior probability of class C
* $P(X)$: Prior probability of predictor X

---

## 🧠 Types of Naive Bayes

### 1️⃣ **Multinomial Naive Bayes**

* Used for **text classification**, especially when features represent **word counts** or frequencies.
* Assumes features follow a **multinomial distribution**.
* Common with **CountVectorizer** or **TF-IDF**.
* Example: Spam classification based on the number of times each word appears.

✔️ Best for: Document classification, NLP, spam detection.

---

### 2️⃣ **Gaussian Naive Bayes**

* Assumes features are **continuous** and follow a **normal (Gaussian) distribution**.
* Not ideal for text data, but works well for datasets with **numeric features** (e.g., height, age).

✔️ Best for: Medical data, biometric data, or any dataset with continuous features.

---

### 3️⃣ **Bernoulli Naive Bayes**

* Assumes **binary feature vectors** (presence or absence of words).
* Each feature is independent and has only 2 values (0 or 1).
* Often used when we care about whether a word is present or not — not how many times it appears.

✔️ Best for: Text classification with binary term occurrence (word present or not).

---

## 🧮 CountVectorizer

* Converts a collection of text documents into a matrix of **token counts**.
* Each row = a document, each column = a token (word).
* Value = number of times a word appears in that document.

🧾 Example:
If you have 3 messages like:

```
1. "I love AI"
2. "AI is fun"
3. "I love coding"
```

Vocabulary = {I, love, AI, is, fun, coding}

Result:

| I | love | AI | is | fun | coding |
| - | ---- | -- | -- | --- | ------ |
| 1 | 1    | 1  | 0  | 0   | 0      |
| 0 | 0    | 1  | 1  | 1   | 0      |
| 1 | 1    | 0  | 0  | 0   | 1      |

---

## 📏 TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF is a text vectorization technique that **reduces the weight** of common words (like "the", "is") and **boosts** the importance of rare but meaningful words.

### 📐 Formula:

$$
TF\text{-}IDF = TF \times IDF
$$

* **TF (Term Frequency)**: Frequency of a term in a document
* **IDF (Inverse Document Frequency)**: Penalizes words that occur in many documents

### ✅ Use:

* Helps identify **important words** in a document.
* Works better than CountVectorizer when **word importance** matters.

---

## 🧮 Summary Table

| Technique       | Description                                  | Best Use Case                              |
| --------------- | -------------------------------------------- | ------------------------------------------ |
| Multinomial NB  | Uses frequency of words (discrete)           | Text classification, spam detection        |
| Gaussian NB     | Uses continuous features (e.g., float data)  | Medical or numeric datasets                |
| Bernoulli NB    | Uses binary features (0/1 for word presence) | Text classification where presence matters |
| CountVectorizer | Converts text into word count vectors        | Bag-of-Words models                        |
| TF-IDF          | Converts text into weighted word importance  | Improved text understanding                |

---