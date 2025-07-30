import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

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
# plt.show()

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
plt.show()

sns.pairplot(df,hue='Target')
plt.show()


# Corelation Matrix
numeric_df=df[['Target','Num_Character','Word_Count','Sent_Count']]
plt.figure(figsize=(10,4))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')
# sesimic,bwr,PiYG,PrGn,RdBu
plt.title('Correlation Heatmap (numerical Feature)')
plt.show()