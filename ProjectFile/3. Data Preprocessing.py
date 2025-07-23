import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter

#print(ps.stem('loving'))
# nltk.download('stopwords')
# print(stopwords.words('english'))
# print(string.punctuation)

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
    # Pie chart
    print("\nTarget Distribution:")
    print(df["Target"].value_counts())
    plt.pie(df["Target"].value_counts(), labels=["ham", "spam"], autopct="%0.2f%%", colors=["skyblue", "lightcoral"])
    plt.title("Distribution of Ham vs Spam Messages")
    plt.axis('equal')  
    plt.show()

    nltk.download('punkt_tab')

    # Feature Engineering
    df['num_characters'] = df['Text'].apply(len)
    df['Word_list'] = df['Text'].apply(lambda x: nltk.word_tokenize(x))
    df['Word_count'] = df['Word_list'].apply(len)
    df['Sent_list'] = df['Text'].apply(lambda x: nltk.sent_tokenize(x))
    df['Sent_count'] = df['Sent_list'].apply(len)

    # Descriptive Stats
    print("\nOverall Stats:")
    print(df[['Text', 'num_characters', 'Word_list', 'Word_count', 'Sent_list', 'Sent_count']].describe())
    
    print("\nHam Message Statistics:")
    print(df[df['Target'] == 0][['num_characters', 'Word_count', 'Sent_count']].describe())
    
    print("\nSpam Message Statistics:")
    print(df[df['Target'] == 1][['num_characters', 'Word_count', 'Sent_count']].describe())

    # Histograms
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(data=df[df['Target'] == 0], x='num_characters', bins=30, color='lightgrey', label='Ham', kde=True, ax=axes[0])
    sns.histplot(data=df[df['Target'] == 1], x='num_characters', bins=30, color='red', label='Spam', kde=True, ax=axes[0])
    axes[0].set_title("Character Count Distribution")
    axes[0].set_xlabel("Number of Characters")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    sns.histplot(data=df[df['Target'] == 0], x='Word_count', bins=30, color='lightblue', label='Ham', kde=True, ax=axes[1])
    sns.histplot(data=df[df['Target'] == 1], x='Word_count', bins=30, color='orange', label='Spam', kde=True, ax=axes[1])
    axes[1].set_title("Word Count Distribution")
    axes[1].set_xlabel("Number of Words")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Pairplot
    sns.pairplot(df[['num_characters', 'Word_count', 'Sent_count', 'Target']], hue='Target')
    plt.show()

    # Correlation Heatmap
    numeric_df = df[['Target', 'num_characters', 'Word_count', 'Sent_count']]
    plt.figure(figsize=(10, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.show()


# Data Preprocessing
# 1) Lower case
# 2) Tokenization
# 3) Removing special characters
# 4) Removing stop words and punctuations
# 5) Stemming


#Example 
# Do if new file
#import nltk
#nltk.download('punkt_tab')

def transform_text(text):
    # Step 1 : Lower Case
    text = text.lower()
    # print("Lowercased:", text)
    
    # Step 2 : Tokenization
    text = nltk.word_tokenize(text)
    # print("Tokenized:", text)
    
    # Step 3 : Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():  
            # This will exclude % and other special characters
            # isalnum => alphanumeric
            y.append(i)
    
    # print("Alphanumeric Only:", y)
    

    # from nltk.corpus import stopwords
    # nltk.download('stopwords')
    # print(stopwords.words('english'))
    # print(string.punctuation)
    # do this to stop what are stop words and punctuations
    
    # Removing Stop words and punctuations
    # this is called cloning [:]
    text = y[:]  # clone y
    y.clear()
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
    
    # print("Without Stopwords:", y)
    
    
    # Stemming
    ps = PorterStemmer()
    text = y[:]
    y.clear()
    for i in text :
        y.append(ps.stem(i))
        
    # print(y)
    # return before stemming = return y
    return " ".join(y)

# output = transform_text("Hi how are you? Kshitij 20% I loved the Offline Session at studio")
# print("Final Output:", output)



# Main Execution Block
df = load_and_clean_data("ProjectFile/spam new.csv")
perform_eda(df)

# Saving the transformed text in a new column
df['Transformed_Text']=df['Text'].apply(transform_text)
print(df.describe())
print(df.info())

# pip install wordcloud
# from wordcloud import WordCloud

# Create WordCloud object
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# Generate word strings
spam_words = df[df['Target'] == 1]['Transformed_Text'].str.cat(sep=" ")
ham_words = df[df['Target'] == 0]['Transformed_Text'].str.cat(sep=" ")

# Generate word clouds
spam_wc = wc.generate(spam_words)
ham_wc = wc.generate(ham_words)

# Plot side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 8))

# Spam WordCloud
axs[0].imshow(spam_wc, interpolation='bilinear')
axs[0].set_title("Spam Messages Word Cloud")
axs[0].axis('off')

ham_wc = WordCloud(
    width=500,
    height=500,
    min_font_size=10,
    background_color='white',
    colormap='plasma'  # Set the color scheme here
).generate(ham_words)

axs[1].imshow(ham_wc, interpolation='bilinear')  # No color/map params here
axs[1].set_title("Ham Messages Word Cloud")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# It returns a list of all transformed text where the message is labeled as spam (Target == 1).
# print(df[df['Target']==1]['Transformed_Text'].tolist())

spam_corpus =[]
for msg in df[df['Target']==1]['Transformed_Text'].tolist():
    for words in msg.split():
        spam_corpus.append(words)

# print(spam_corpus)
# print(len(spam_corpus))

#from collections import Counter
# print(Counter(spam_corpus).most_common(30))

# Get top 30 most common words in spam
top_30_spam = Counter(spam_corpus).most_common(30)
df_top_spam = pd.DataFrame(top_30_spam, columns=["Word", "Frequency"])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x="Word", y="Frequency", data=df_top_spam, palette="magma")
plt.xticks(rotation="vertical")
plt.title("Top 30 Words in Spam Corpus")
plt.tight_layout()
plt.show()

# Create ham corpus
ham_corpus = []
for msg in df[df['Target'] == 0]['Transformed_Text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

# Get top 30 most common words in ham
top_30_ham = Counter(ham_corpus).most_common(30)
df_top_ham = pd.DataFrame(top_30_ham, columns=["Word", "Frequency"])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x="Word", y="Frequency", data=df_top_ham, palette="viridis")
plt.xticks(rotation="vertical")
plt.title("Top 30 Words in Ham Corpus")
plt.tight_layout()
plt.show()


""" --- Plot Side-by-Side ---

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

sns.barplot(x="Word", y="Frequency", data=df_top_spam, palette="magma", ax=axes[0])
axes[0].set_title("Top 30 Words in Spam Messages")
axes[0].tick_params(axis='x', rotation=90)

sns.barplot(x="Word", y="Frequency", data=df_top_ham, palette="viridis", ax=axes[1])
axes[1].set_title("Top 30 Words in Ham Messages")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

"""