import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

# Ensure required NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Function to load and clean the data
def load_and_clean_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
    df.rename(columns={'v1': 'Target', 'v2': 'Text'}, inplace=True)
    
    encoder = LabelEncoder()
    df['Target'] = encoder.fit_transform(df['Target'])

    print("\nNull values per column:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

    df.drop_duplicates(keep="first", inplace=True)
    print("Shape after cleaning:", df.shape)
    
    return df

# Function for EDA
def perform_eda(df):
    print("\nTarget Distribution:")
    print(df["Target"].value_counts())
    plt.pie(df["Target"].value_counts(), labels=["ham", "spam"], autopct="%0.2f%%", colors=["skyblue", "lightcoral"])
    plt.title("Distribution of Ham vs Spam Messages")
    plt.axis('equal')  
    #plt.show()

    # Feature Engineering
    df['num_characters'] = df['Text'].apply(len)
    df['Word_list'] = df['Text'].apply(lambda x: nltk.word_tokenize(x))
    df['Word_count'] = df['Word_list'].apply(len)
    df['Sent_list'] = df['Text'].apply(lambda x: nltk.sent_tokenize(x))
    df['Sent_count'] = df['Sent_list'].apply(len)

    print("\nOverall Stats:")
    print(df[['Text', 'num_characters', 'Word_list', 'Word_count', 'Sent_list', 'Sent_count']].describe())
    
    print("\nHam Message Statistics:")
    print(df[df['Target'] == 0][['num_characters', 'Word_count', 'Sent_count']].describe())
    
    print("\nSpam Message Statistics:")
    print(df[df['Target'] == 1][['num_characters', 'Word_count', 'Sent_count']].describe())

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(data=df[df['Target'] == 0], x='num_characters', bins=30, color='lightgrey', label='Ham', kde=True, ax=axes[0])
    sns.histplot(data=df[df['Target'] == 1], x='num_characters', bins=30, color='red', label='Spam', kde=True, ax=axes[0])
    axes[0].set_title("Character Count Distribution")
    axes[0].legend()

    sns.histplot(data=df[df['Target'] == 0], x='Word_count', bins=30, color='lightblue', label='Ham', kde=True, ax=axes[1])
    sns.histplot(data=df[df['Target'] == 1], x='Word_count', bins=30, color='orange', label='Spam', kde=True, ax=axes[1])
    axes[1].set_title("Word Count Distribution")
    axes[1].legend()

    plt.tight_layout()
    #plt.show()

    sns.pairplot(df[['num_characters', 'Word_count', 'Sent_count', 'Target']], hue='Target')
    #plt.show()

    plt.figure(figsize=(10, 4))
    sns.heatmap(df[['Target', 'num_characters', 'Word_count', 'Sent_count']].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap (Numerical Features)")
    #plt.show()

# Function for spam & ham word analysis
def run_spam_analysis(df):
    ps = PorterStemmer()

    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = [i for i in text if i.isalnum()]
        y = [i for i in y if i not in stopwords.words('english')]
        y = [ps.stem(i) for i in y]
        return " ".join(y)

    df['Transformed_Text'] = df['Text'].apply(transform_text)

    # Word Clouds
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    spam_wc = wc.generate(df[df['Target'] == 1]['Transformed_Text'].str.cat(sep=" "))
    ham_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white', colormap='plasma')\
        .generate(df[df['Target'] == 0]['Transformed_Text'].str.cat(sep=" "))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(spam_wc, interpolation='bilinear')
    axs[0].set_title("Spam Word Cloud")
    axs[0].axis('off')
    axs[1].imshow(ham_wc, interpolation='bilinear')
    axs[1].set_title("Ham Word Cloud")
    axs[1].axis('off')
    plt.tight_layout()
    #plt.show()

    # Corpus
    spam_corpus = [word for msg in df[df['Target'] == 1]['Transformed_Text'] for word in msg.split()]
    ham_corpus = [word for msg in df[df['Target'] == 0]['Transformed_Text'] for word in msg.split()]
    df_top_spam = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=["Word", "Frequency"])
    df_top_ham = pd.DataFrame(Counter(ham_corpus).most_common(30), columns=["Word", "Frequency"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    sns.barplot(x="Word", y="Frequency", data=df_top_spam, palette="magma", ax=axes[0])
    axes[0].set_title("Top 30 Spam Words")
    axes[0].tick_params(axis='x', rotation=90)

    sns.barplot(x="Word", y="Frequency", data=df_top_ham, palette="viridis", ax=axes[1])
    axes[1].set_title("Top 30 Ham Words")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    #plt.show()

#  Run Everything
df = load_and_clean_data("ProjectFile/spam new.csv")
perform_eda(df)
run_spam_analysis(df)


# Model Building 

# from sklearn.feature_extraction.text import CountVectorizer
# Vectorization
cv = CountVectorizer()
x = cv.fit_transform(df['Transformed_Text']).toarray()
print("Feature Matrix:\n", x)
print("Shape:", x.shape)

# Target Variable
y = df['Target'].values
print("Target Labels:\n", y)

# Train-Test Split (corrected typo: text_size ➝ test_size)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
# from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

# Models
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Train and Predict with GaussianNB
gnb.fit(x_train, y_train)
y_pred1 = gnb.predict(x_test)

# Train and Predict with MultinomialNB
mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)

# Train and Predict with BernoulliNB
bnb.fit(x_train, y_train)
y_pred3 = bnb.predict(x_test)
