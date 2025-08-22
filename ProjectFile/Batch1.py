import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk .corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix
import os 
import pickle

df = pd.read_csv("ProjectFile/spam.csv",encoding='latin-1')
# print(df)

# Drop Last three Columns 
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

# print(df.head())

# Rename Column 
# print()
# print("Rename Columns")
df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
# print(df.head())

# Refitting First column Traget
# print()
# print("Refitting the Data in Column = Target")
# print(df['Target'].unique())

# from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])
# print(df.head())

# Check null values or missing values 
# print()
# print("Check null values or missing values ")
# print(df.isnull().sum())

# Check duplicate Values
# print()
# print("Check Duplicate Values")
# print(df.duplicated().sum())

# Remove Duplicate Values 
# print()
# print("Remove Duplicate Values")
df.drop_duplicates(keep='first')
# print(df.head())

# EDA

# Reading Target Data 
# print()
# print("Reading Target Data ")
# print(df['Target'].value_counts())

# Shape of Dataset
# print()
# print("Shape of Dataset")
# print(df['Target'].shape)
# print(df.shape)

# Pie chart 
# import matplotlib.pyplot as plt
plt.pie(df['Target'].value_counts(),labels=['Ham','Spam'],autopct="%0.2f%%",colors=['skyblue','lightcoral'])
plt.legend()
plt.title("Distribution of Ham Vs Spam")
plt.axis('equal')
# plt.show()

# pip install nltk 
# import nltk
# nltk.download('punkt_tab')

# Character Count

df['num_characters']=df['Text'].apply(len)
# print(df['num_characters'])

# Word Tokenization
# print()
df['Word_List']=df['Text'].apply(lambda x : nltk.word_tokenize(x))
# print(df.head())

# Word Count 
# print()
df['Word_Count']=df['Word_List'].apply(len)
# print(df.head())

# Sentence Tokenization
# print()
df['Sent_List']=df['Text'].apply(lambda x : nltk.sent_tokenize(x))
# print(df.head())

# Sentence Count 
# print()
df['Sent_Count']=df['Sent_List'].apply(len)
# print(df.head())

# Quick Preview

# Head and Describe 
# print()
# print(df[['Text','num_characters','Word_List','Word_Count','Sent_List','Sent_Count']].describe())

# Analysis for Ham Messages 
# print()
# print("Analysis for Ham Messages ")
# print(df[df['Target']==0][['num_characters','Word_Count','Sent_Count']].describe())


# Analysis for Spam Messages 
# print()
# print("Analysis for Spam Messages ")
# print(df[df['Target']==1][['num_characters','Word_Count','Sent_Count']].describe())


# import seaborn as sns
# Set Style
sns.set_style(style='whitegrid')

# Create Figure for sub plots
fig,axes=plt.subplots(1,3,figsize=(12,5))

# Histogram for Character Count
sns.histplot(data=df[df['Target']==0],x='num_characters',bins=30,color='grey',label='Ham',kde=True,ax=axes[0])
sns.histplot(data=df[df['Target']==1],x='num_characters',bins=30,color='red',label='Spam',kde=True,ax=axes[0])

axes[0].set_title("Character Count Distribution")
axes[0].set_xlabel('Number of Characters')
axes[0].set_ylabel('Frequency')
axes[0].legend()


# Histogram for Word Count
sns.histplot(data=df[df['Target']==0],x='Word_Count',bins=30,color='blue',label='Ham',kde=True,ax=axes[1])
sns.histplot(data=df[df['Target']==1],x='Word_Count',bins=30,color='orange',label='Spam',kde=True,ax=axes[1])

axes[1].set_title("Character Count Distribution")
axes[1].set_xlabel('Number of Words')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Histogram for Word Count
sns.histplot(data=df[df['Target']==0],x='Sent_Count',bins=30,color='green',label='Ham',kde=True,ax=axes[2])
sns.histplot(data=df[df['Target']==1],x='Sent_Count',bins=30,color='black',label='Spam',kde=True,ax=axes[2])

axes[2].set_title("Character Count Distribution")
axes[2].set_xlabel('Number of Sentences')
axes[2].set_ylabel('Frequency')
axes[2].legend()


plt.tight_layout()
#plt.show()


# Pair Plot
sns.pairplot(df,hue='Target')
#plt.show()

# Select only numeric columns for correlation 

numeric_df=df[['Target','num_characters','Word_Count','Sent_Count']]
plt.figure(figsize=(10,5))
sns.heatmap(numeric_df.corr(),annot=True,cmap='PiYG')
# bwr,sesimic,PiYG,PrGn,RdBu
plt.title("Correlation Heatmap (Numeric Features)")
# plt.show()


# Data Preprocessing 


def transform_text(text):
    
    # print()
    # print(f"Text : {text}")
    # Step 1 : Lower Case
    # print()
    text = text.lower()
    # print(f"Lower Case : {text}")
    
    # Step 2 : Tokenization
    # print()
    text = nltk.word_tokenize(text)
    # print(f"Tokenization : {text}")
    
    # Step 3 : Removing Special Chars and Punctuation
    # print()
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    # print(f"Alphahumeric Only : {y}")
    
    # Step 4 : Stopwords 
    # from nltk .corpus import stopwords
    # nltk.download('stopwords')
    # print(stopwords.words('english'))
    # import string
    # print(string.punctuation)
    
    # print()
    text = y [:]
    y.clear()
    for i in text : 
        if i not in stopwords.words('english'):
            y.append(i)
            
    # print(f"After Removing Stopwords : {y}")
    
    
    # Step 5 : Stemming 
    # from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    # print()
    text = y [:]
    y.clear()
    for i in text : 
        y.append(ps.stem(i))
    
    # print(f"After Stemming {y}")
        
    return " ".join(y)

# transform_text("Thomas Loves Driving  a Car ! & Mary Loves Eating a Pizza ? ")
print()
df['Transformed_Text']=df['Text'].apply(transform_text)
print(df.describe())
print()
print(df.info())
print()


# pip install wordCloud
# from wordcloud import WordCloud

# Create a Word Cloud Object 
spam_wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
ham_wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white',colormap='plasma')

# Genrate word cloud string 
spam_words = df[df['Target']==1]['Transformed_Text'].str.cat(sep=' ')
ham_words = df[df['Target']==0]['Transformed_Text'].str.cat(sep=' ')

# genrate Word cloud 
spam_gwc = spam_wc.generate(spam_words)
ham_gwc = ham_wc.generate(ham_words)

# plot side by side 
fig,axs=plt.subplots(1,2,figsize=(10,8))

# Spam Word Cloud
axs[0].imshow(spam_gwc,interpolation='bilinear')
axs[0].set_title('Spam Messages Word Cloud')
axs[0].axis('off')

# Ham Word Cloud

axs[1].imshow(ham_gwc,interpolation='bilinear')
axs[1].set_title('Ham Messages Word Cloud')
axs[1].axis('off')

plt.tight_layout()
# plt.show()



        
# print()
# print(f"{spam_corpus}")

# print()
# print(f"Length : {len(spam_corpus)}")

from collections import Counter
# print(Counter(spam_corpus).most_common(30))



#plt.figure(figsize=(12,6))
#sns.barplot(x='Words',y='Frequency',data=df_top_spam,palette='magma')
#plt.xticks(rotation='vertical')
#plt.title("Top 30 Words in Spam Corpus")
#plt.tight_layout()
# plt.show()

# print(df[df['Target']==1]['Transformed_Text'].tolist())
spam_corpus=[]
for msg in df[df['Target']==1]['Transformed_Text'].tolist():
    for words in msg.split():
        spam_corpus.append(words)

top_30_spam = Counter(spam_corpus).most_common(30)
df_top_spam = pd.DataFrame(top_30_spam,columns=['Words','Frequency'])

ham_corpus=[]
for msg in df[df['Target']==0]['Transformed_Text'].tolist():
    for words in msg.split():
        ham_corpus.append(words)
        
top_30_ham = Counter(ham_corpus).most_common(30)
df_top_ham = pd.DataFrame(top_30_ham,columns=['Words','Frequency'])

fig,axs = plt.subplots(1,2,figsize=(10,6),sharey=True)
sns.barplot(x='Words',y='Frequency',data=df_top_spam,palette='magma',ax=axs[0])
axs[0].set_title("Top 30 Words in Spam Corpus")
axs[0].tick_params(axis='x',rotation=90)

sns.barplot(x='Words',y='Frequency',data=df_top_ham,palette='viridis',ax=axs[1])
axs[1].set_title("Top 30 Words in Ham Corpus")
axs[1].tick_params(axis='x',rotation=90)

plt.tight_layout()
# plt.show()

# Model Building 

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split

cv = CountVectorizer()
x = cv.fit_transform(df['Transformed_Text']).toarray()
print(f"Feature Matrix : \n {x}")
print(f"Shape : {x.shape}")

y = df['Target'].values
print(f'Target Labels:\n {y}')
# Train-Test-Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

# from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# from sklearn.metrics import accuracy_score, precision_score,confusion_matrix
# Train and Predict with GaussianNB
print()
gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print('GaussianNB')
print(f'Confusion Matrix :\n {confusion_matrix(y_test,y_pred1)}')
print(f'Accuracy Score : {accuracy_score(y_test,y_pred1)}')
print(f'Precision Score : {precision_score(y_test,y_pred1)}')
print()

# Train and Predict with MultinomialNB
print()
mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print('MultinomialNB')
print(f'Confusion Matrix :\n {confusion_matrix(y_test,y_pred2)}')
print(f'Accuracy Score : {accuracy_score(y_test,y_pred2)}')
print(f'Precision Score : {precision_score(y_test,y_pred2)}')
print()

# Train and Predict with BernoulliNB
print()
bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print('BernoulliNB')
print(f'Confusion Matrix :\n {confusion_matrix(y_test,y_pred3)}')
print(f'Accuracy Score : {accuracy_score(y_test,y_pred3)}')
print(f'Precision Score : {precision_score(y_test,y_pred3)}')
print()


# Import all the models 
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              StackingClassifier)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score,precision_score,confusion_matrix

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Transformed_Text']).toarray()
y = df['Target'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

models = {"GaussianNB":GaussianNB(),
          "MultinomialNB":MultinomialNB(),
          "BernoulliNB":BernoulliNB(),
          "Logistic Regression":LogisticRegression(solver='liblinear'),
          "Support Vector Classifier":SVC(),
          "Decision Tree Classifier":DecisionTreeClassifier(),
          "K Neighbors Classifier":KNeighborsClassifier(),
          "Random Forest Classifier":RandomForestClassifier(n_estimators=100),
          "Ada Boost Classifier":AdaBoostClassifier(n_estimators=50),
          "Extra Trees Classifier":ExtraTreesClassifier(n_estimators=100),
          "Gradient Boosting Classifier":GradientBoostingClassifier(),
          }

performance_data = []

for name , model in models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred)
    confusion_mtx = confusion_matrix(y_test,y_pred)
    print(f"\n {name}")
    print(f"Accuracy Score : {acc}")
    print(f"Precision Score : {prec}")
    print(f"Confusion Matrix :\n {confusion_mtx}")
    performance_data.append((name,acc,prec))
    
print()
print(performance_data)    



# Adding Stacking Classifier 
stack = StackingClassifier(estimators=[('mnb',MultinomialNB()),
                                       ('lr',LogisticRegression(solver='liblinear'))],
                           final_estimator=LogisticRegression())

stack.fit(x_train,y_train)
y_pred_stack = stack.predict(x_test)
acc_stack = accuracy_score(y_test,y_pred_stack)
prec_stack = precision_score(y_test,y_pred_stack)
confusion_mtx_stack = confusion_matrix(y_test,y_pred_stack)
print(f"\n {name}")
print(f"Accuracy Score : {acc_stack}")
print(f"Precision Score : {prec_stack}")
print(f"Confusion Matrix :\n {confusion_mtx_stack}")
performance_data.append(('Stacking Classifier',acc_stack,prec_stack))
print(performance_data) 

performance_data = pd.DataFrame(performance_data,columns=['Algorithm','Accuracy','Precision'])
performance_data_melted = pd.melt(performance_data,id_vars='Algorithm',
                                   var_name = 'Metric',
                                   value_name='Score')


sns.catplot(x = 'Algorithm',
            y = 'Score',
            hue = 'Metric',
            data = performance_data_melted,
            kind = 'bar',
            height = 6)

plt.xticks(rotation=45,ha='right')
plt.ylim(0.5,1.05)
plt.title('Model Comparison - Accuracy v/s Precision')
plt.tight_layout()
plt.show()

# import os 
# import pickle

os.makedirs("models2",exist_ok=True)
pickle.dump(tfidf,open('models2/vectorizer.pkl','wb'))
pickle.dump(stack,open('models2/model.pkl','wb'))

print(f'Model installed Successfully')