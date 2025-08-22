
#  Machine Learning – Domain Internship

A structured learning-based repository containing a step-by-step, end-to-end walkthrough of the machine learning lifecycle—from Python fundamentals and data preprocessing to model training and deployment.

---

### Overview of Contents

* **Python Foundations**

  * Core concepts such as `Python basics`, fundamental data types, loops, functions, and error handling.

* **Data Manipulation & Visualization**

  * In-depth use of libraries including:

    * **Pandas**: Series, DataFrame operations, filtering, grouping, merging.
    * **Matplotlib** & **Seaborn**: Plotting essentials—line charts, histograms, scatter plots, heatmaps, pairplots.

* **Complete ML Workflow**

  * **Data Extraction & Loading**: Reading CSVs, Excel, SQL connectors.
  * **Exploratory Data Analysis (EDA)**: Statistical summaries, distribution analysis, correlations, visualizations.
  * **Data Cleaning**: Handling missing data, duplicates, outliers, and data normalization.
  * **Feature Engineering & Preprocessing**: Encoding categorical variables, scaling, and transforming features.
  * **Model Development**: Training models like Logistic Regression, Decision Trees, Naive Bayes; evaluating their performance.
  * **Model Persistence**: Saving trained models and vectorizers with `pickle` for reuse.

---

### Structured Workflow (Code Files Sequence)

1. **Data\_Cleaning**
   Implements critical data hygiene routines—null value imputation, deduplication, and error handling.

2. **EDA**
   Visualizes dataset features, distributions, correlations, and general statistics for better insight.

3. **Data\_Preprocessing**
   Prepares data for modeling: encoding, scaling, vectorizing, and processing text features if any.

4. **Model\_Building**
   Trains models, evaluates on validation/test set using metrics like accuracy, precision, recall, and saves final models (`model.pkl`, `vectorizer.pkl`) for deployment.

---

### Purpose & Value

* **Comprehensive Learning Guide**: Documents every step of progressing from Python basics to end-to-end ML model deployment.
* **Progression in Abstraction**: Gradually builds from raw data handling to sophisticated modeling workflows.
* **Reusable Components**: Each script serves as a standalone piece—adaptable, modular, and easy to understand.

---

# 📧 Email/SMS Spam Classifier

An interactive machine learning web application that detects whether a given text message (email or SMS) is **Spam** or **Not Spam (Ham)**. Built using Python, trained on real-world labeled data, and deployed locally via **Streamlit**.

> 🎯 **Goal**: Empower users to test spam detection in real-time using natural language processing and machine learning.

<img width="1095" height="709" alt="Image" src="https://github.com/user-attachments/assets/9a78ff17-c0b7-45fc-9f0e-51470bde32c1" />

---

## 📁 Folder Structure

```
📦 Email-SMS-Spam-Classifier/
├── 📁 models2/
│   ├── app2.py               # Streamlit UI logic and model inference
│   ├── model.pkl             # Trained ML model (pickle)
│   └── vectorizer.pkl        # Text vectorizer (TF-IDF or CountVectorizer)
│
├── 📁 ProjectFile/
│   └── 📁 models/
│       ├── 1.Data_Cleaning.py
│       ├── 2.EDA.py
│       ├── 3. Data Preprocessing.py
│       └── 4. Model Building.py
│       ├── model.pkl
│       └── vectorizer.pkl
│
├── 📁 Project Graphs/
│   ├── Correlation.png
│   ├── Distribution Plot.png
│   ├── Histogram.png
│   ├── Word Cloud.png
│   ├── Top 30 Words.png
│   └── Model Comparison.png
```

---

## 🛠️ Tech Stack

| Tool                     | Purpose                     |
| ------------------------ | --------------------------- |
| **Python**               | Programming Language        |
| **Pandas**               | Data manipulation           |
| **Scikit-learn**         | ML Algorithms + Metrics     |
| **Matplotlib / Seaborn** | Data Visualization          |
| **Streamlit**            | Frontend UI for Predictions |
| **Pickle**               | Model and vectorizer export |

---

## 🧪 How to Run Locally


#### Clone repo
````py
git clone https://github.com/KshitijSawant1/Machine-Learning---Domain-Internship.git
cd Machine-Learning---Domain-Internship
````
#### Explore ML pipeline scripts
````
cd ProjectFile/models
# Run each script sequentially:
python 1.Data_Cleaning.py
python 2.EDA.py
python 3.Data_Preprocessing.py
python 4.Model_Building.py
````

#### Run the demo interface (assuming dependencies are installed)
````
cd ../../models2
streamlit run app2.py
````

---

## 📚 Learning Outcomes

✅ Hands-on experience with:

* Real-world spam datasets
* NLP and feature extraction
* Model performance comparison
* Web UI development using Streamlit
* Model serialization and deployment pipeline

---

## 🧠 Future Improvements

* Deploy on public server (e.g., HuggingFace Spaces / Heroku)
* Add LSTM or BERT-based spam detection
* Save message prediction history
* Support multiple languages



---

##  Project Structure

``` Machine-Learning---Domain-Internship/
├── ProjectFile/
│   └── models/
│       ├── 1.Data\_Cleaning.py
│       ├── 2.EDA.py
│       ├── 3.Data\_Preprocessing.py
│       └── 4.Model\_Building.py
│
├── models2/
│   ├── app2.py               # Streamlit-based UI for prediction demo
│   ├── model.pkl             # Saved trained model
│   └── vectorizer.pkl        # Saved feature vectorizer
│
├── Project Graphs/
│   ├── Correlation.png
│   ├── Distribution Plot.png
│   ├── Histogram.png
│   └── Word Cloud.png
│
└── README.md

````

---

##  What’s Inside / Learning Path

### 1. Python Basics & Data Libraries
- Fundamentals of Python programming: variables, loops, functions, error handling.
- Hands-on usage of `pandas` for data manipulation, and `matplotlib` + `seaborn` for visual analytics.

### 2. Data Preprocessing & Analysis
- `1.Data_Cleaning.py`: Handling missing values, outliers, duplicates.
- `2.EDA.py`: Exploring data distributions, correlations, and generating insight-focused visuals.

### 3. Feature Engineering & Preparation
- `3.Data_Preprocessing.py`: Encoding categorical data, scaling numerical features, and text vectorization.

### 4. Model Development Lifecycle
- `4.Model_Building.py`: Training machine learning models (e.g., logistic regression, Naive Bayes), evaluating performance, and serializing: `model.pkl`, `vectorizer.pkl`.

---

##  Demo Interface (Streamlit App)

- Live predictive UI using **Streamlit**, located in `models2/app2.py`.
- Allows users to input text and view spam/ham classification results in real-time with the trained model.

---

##  Installation & Execution Guide


---

## Visual Insights

Check out the `Project Graphs/` folder for generated visual outputs:

* Correlation heatmaps
* Distribution histograms
* Word cloud visualizations, among others

---

## Technologies Used

* **Python** — Core language for scripting
* `pandas`, `numpy` — Data handling
* `matplotlib`, `seaborn` — Visualization
* `scikit-learn` — Model building and evaluation
* **Streamlit** — Easy deployment of frontend UI
* `pickle` — Model persistence

---

## Benefits of This Repository

* **Educational pipeline structure** — Great reference for learning or teaching ML fundamentals.
* **Modular and replicable** — Each script tackles one stage of the ML workflow.
* **Interactive demo experience** — Immediate feedback via Streamlit UI.

---

## Author

**Kshitij Sawant**
GitHub: [@KshitijSawant1](https://github.com/KshitijSawant1)
Reach out via GitHub for feedback, suggestions, or project deployment assistance.

---