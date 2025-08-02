import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string 
from nltk.stem.porter import PorterStemmer 
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('ProjectFile/spam.csv',encoding='latin-1')
print(df.head())

# Drop Last Three Columns 
print()
print("Drop Last Three Columns")
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
print(df.head())

# Rename Columns 
print()
print("Rename Columns ")
df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
print(df.head())

# Refitting first column "Target"
print()
print("Checking Unique Values in Column Target")
print(df['Target'].unique())
print()
print("Refitting First Columns Target")

# from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])
print(df.head())

# Check for Missing Values
print()
print("Check for Missing Values")
print(df.isnull().sum())

# Check for Duplicate Values
print()
print("Check for Duplicate Values")
print(df.duplicated().sum())

# Dropping Duplicated Values
print()
print("Dropping Duplicated Values")
df.drop_duplicates(keep="first")

# Final Shape of Dataset

print()
print("Final Shape of Dataset Column : Target")
print(df['Target'].shape)
print()
print("Final Shape of Dataset")
print(df.shape)
print()
print("Final Info of Dataset")
print(df.info())
print()


# import matplotlib.pyplot as plt
# import seaborn as sns

# Data Reading Target
print()
print("Data Reading Target")
print(df['Target'].value_counts())


# Pie Chart 
plt.pie(df['Target'].value_counts(),labels=['Ham','Spam'],autopct='%0.2f%%',colors=['lightblue','lightcoral'])
plt.title("Distribution of Ham vs Spam")
plt.legend()
# #plt.show()

# pip install nltk
# import nltk
# nltk.download('punkt_tab')

# Character Count 
print()
print("Character Count ")
df['Num_Character']=df['Text'].apply(len)
print(df.head())

# Word Tokenization
print()
print('Word Tokenization')
df['Word_List']=df['Text'].apply(lambda x : nltk.word_tokenize(x))

# Word Count
print()
print("Word Count")
df['Word_Count']=df['Word_List'].apply(len)

# Sentence Tokenization
print()
print('Sentence Tokenization')
df['Sent_List']=df['Text'].apply(lambda x : nltk.sent_tokenize(x))

# Sentence Count
print()
print("Sentence Count")
df['Sent_Count']=df['Sent_List'].apply(len)


# Quick Preview
# Head and Describe
print(df[['Text','Num_Character','Word_List','Word_Count','Sent_List','Sent_Count']].describe())

# Analysis for Ham Messages [Target==0]
print()
print("Ham Messages Statistics")
print(df[df['Target']==0][['Text','Num_Character','Word_Count','Sent_Count']].describe())

# Analysis for Spam Messages [Target==1]
print()
print("Spam Messages Statistics")
print(df[df['Target']==1][['Text','Num_Character','Word_Count','Sent_Count']].describe())

# Create Figure with 3 Subplots
fig,axes = plt.subplots(1,3,figsize=(12,4))

# 1) Histogram for Character Count
sns.histplot(data=df[df['Target']==0],x='Num_Character',bins=30,color='lightgrey',label='Ham',
             kde=True,ax=axes[0])
sns.histplot(data=df[df['Target']==1],x='Num_Character',bins=30,color='red',label='Spam',
             kde=True,ax=axes[0])
axes[0].set_title("Character Count Distribution")
axes[0].set_xlabel("Number of Character")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# 2) Histogram for Word Count
sns.histplot(data=df[df['Target']==0],x='Word_Count',bins=30,color='lightblue',label='Ham',
             kde=True,ax=axes[1])
sns.histplot(data=df[df['Target']==1],x='Word_Count',bins=30,color='orange',label='Spam',
             kde=True,ax=axes[1])
axes[1].set_title("Word Count Distribution")
axes[1].set_xlabel("Word of Character")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# 3) Histogram for Sentence Count
sns.histplot(data=df[df['Target']==0],x='Sent_Count',bins=30,color='lightgreen',label='Ham',
             kde=True,ax=axes[2])
sns.histplot(data=df[df['Target']==1],x='Sent_Count',bins=30,color='black',label='Spam',
             kde=True,ax=axes[2])
axes[2].set_title("Sent Count Distribution")
axes[2].set_xlabel("Sentence of Character")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
#plt.show()

sns.pairplot(df,hue='Target')
#plt.show()


# Corelation Matrix
numeric_df=df[['Target','Num_Character','Word_Count','Sent_Count']]
plt.figure(figsize=(10,4))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')
# sesimic,bwr,PiYG,PrGn,RdBu
plt.title('Correlation Heatmap (numerical Feature)')
#plt.show()


# Data Preprocessing 


def transform_text(text):
    
    # print()
    # print(text)
    
    # Step 1 : LowerCase
    text = text.lower()
    # print(f"Lower Cased : {text}")
    
    # Step 2 : Tokenization 
    # print()
    text = nltk.word_tokenize(text)
    # print(f"Tokenized : {text}")
    
    
    #Step 3 : Removing Special Charcaters
    # print()
    y = []
    for i in text :
        if i.isalnum():
            y.append(i)
            # this will put special charters in y
    # print(f"Alphanumeric Only : {y}")
    
    
    # from nltk.corpus import stopwords
    # nltk.download('stopwords')
    # print(stopwords.words('english'))
    # import string
    # print(string.punctuation)
    
    # Removing Stopwords and punctuation
    # print()
    text = y[:] # cloning 
    y.clear()
    for i in text :
        if i not in stopwords.words('english'):
            y.append(i)
    
    # print(f"Without Stopwords : {y}")
    
    # Stemming
    # from nltk.stem.porter import PorterStemmer
    #print()
    text = y[:]
    ps = PorterStemmer()
    y.clear()
    for i in text :
        y.append(ps.stem(i))
        
    return " ".join(y)
    
# output = transform_text("Hi Hello How Are You. @ ? ! / Raj Drives a Car . Mary loves eating Pizza.")
# print(f"Final Output : {output}")

# Trasformed Text Execution 
df['Trasformed_Text']=df['Text'].apply(transform_text)
print(df.describe())
print(df.info())


# pip install wordcloud
# from wordcloud import WordCloud

wc = WordCloud(width= 500, height=500, min_font_size=1, background_color='white')


# Genrate Word Strings 
spam_words = df[df['Target']==1]['Trasformed_Text'].str.cat(sep=" ")
ham_words = df[df['Target']==0]['Trasformed_Text'].str.cat(sep=" ")

# Genrate Word Clouds
spam_wc = wc.generate(spam_words)
ham_wc = wc.generate(ham_words)

fig, axs = plt.subplots(1,2,figsize=(10,8))

# Spam Word Cloud
axs[0].imshow(spam_wc,interpolation='bilinear')
axs[0].set_title('Spam Message Word Cloud')
axs[0].axis('off')

# Ham Word Cloud 
ham_wc = WordCloud(width=500,
                   height=500,
                   min_font_size=10,
                   background_color='white',
                   colormap='plasma').generate(ham_words)

axs[1].imshow(ham_wc,interpolation='bilinear')
axs[1].set_title('Ham Message Word Cloud')
axs[1].axis('off')

plt.tight_layout()
# plt.show()


# Spam Corpus 
spam_corpus = []
for msg in df[df['Target']==1]['Trasformed_Text'].tolist():
    for words in msg.split():
        spam_corpus.append(words)
        
# print(spam_corpus)
# print()
# print(len(spam_corpus))


# from collections import Counter
# print()
# print(Counter(spam_corpus).most_common(30))

top_30_spam = Counter(spam_corpus).most_common(30)

df_top_spam = pd.DataFrame(top_30_spam,columns=['Word','Frequency'])

ham_corpus = []
for msg in df[df['Target']==0]['Trasformed_Text'].tolist():
    for words in msg.split():
        ham_corpus.append(words)
        
top_30_ham = Counter(ham_corpus).most_common(30)

df_top_ham = pd.DataFrame(top_30_ham,columns=['Word','Frequency'])


# plot 
plt.figure(figsize=(12,6))
sns.barplot(x="Word",y='Frequency',data=df_top_spam)
plt.xticks(rotation='vertical')
plt.title("Top 30 Spam Words in Corpus")
plt.tight_layout()
# plt.show()


fig ,axes = plt.subplots(1,2,figsize=(10,6),sharey=True)
sns.barplot(x="Word",y='Frequency',data=df_top_spam,ax=axes[0],palette="magma")
axes[0].set_title("Top 30 Spam Words in Corpus")
axes[0].tick_params(axis='x',rotation=90)

sns.barplot(x="Word",y='Frequency',data=df_top_ham,ax=axes[1],palette="viridis")
axes[1].set_title("Top 30 Ham Words in Corpus")
axes[1].tick_params(axis='x',rotation=90)

plt.tight_layout()
# plt.show()


# Model Building

# from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(df['Trasformed_Text']).toarray()
print()
print(f"Feature Matrix : {x}")
print()
print(f"Shape : {x.shape}")

# Target Values 
print()
y = df['Target'].values
print(f"Target Labels : {y}")


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Train - Test Splt 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

# Models 
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
# Train and Pedict

# GaussianNB
gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print()
print("GaussianNB")
print(f"Accuracy Score : {accuracy_score(y_test,y_pred1)}")
print(f"Confusion Matrix :\n {confusion_matrix(y_test,y_pred1)}")
print(f"Precision Score : {precision_score(y_test,y_pred1)}")
print()


# MultinomialNB
mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print()
print("MultinomialNB")
print(f"Accuracy Score : {accuracy_score(y_test,y_pred2)}")
print(f"Confusion Matrix :\n {confusion_matrix(y_test,y_pred2)}")
print(f"Precision Score : {precision_score(y_test,y_pred2)}")
print()


# BernoulliNB
bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print()
print("BernoulliNB")
print(f"Accuracy Score : {accuracy_score(y_test,y_pred3)}")
print(f"Confusion Matrix :\n {confusion_matrix(y_test,y_pred3)}")
print(f"Precision Score : {precision_score(y_test,y_pred3)}")
print()
