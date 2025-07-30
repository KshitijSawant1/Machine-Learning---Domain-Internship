import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns

df = pd.read_csv("ProjectFile/spam.csv",encoding='latin-1')
print(df)

# Drop Last three Columns 
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

print(df.head())

# Rename Column 
print()
print("Rename Columns")
df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
print(df.head())

# Refitting First column Traget
print()
print("Refitting the Data in Column = Target")
print(df['Target'].unique())

# from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])
print(df.head())

# Check null values or missing values 
print()
print("Check null values or missing values ")
print(df.isnull().sum())

# Check duplicate Values
print()
print("Check Duplicate Values")
print(df.duplicated().sum())

# Remove Duplicate Values 
print()
print("Remove Duplicate Values")
df.drop_duplicates(keep='first')
print(df.head())

# EDA

# Reading Target Data 
print()
print("Reading Target Data ")
print(df['Target'].value_counts())

# Shape of Dataset
print()
print("Shape of Dataset")
print(df['Target'].shape)
print(df.shape)

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
print(df['num_characters'])

# Word Tokenization
print()
df['Word_List']=df['Text'].apply(lambda x : nltk.word_tokenize(x))
print(df.head())

# Word Count 
print()
df['Word_Count']=df['Word_List'].apply(len)
print(df.head())

# Sentence Tokenization
print()
df['Sent_List']=df['Text'].apply(lambda x : nltk.sent_tokenize(x))
print(df.head())

# Sentence Count 
print()
df['Sent_Count']=df['Sent_List'].apply(len)
print(df.head())

# Quick Preview

# Head and Describe 
print()
print(df[['Text','num_characters','Word_List','Word_Count','Sent_List','Sent_Count']].describe())

# Analysis for Ham Messages 
print()
print("Analysis for Ham Messages ")
print(df[df['Target']==0][['num_characters','Word_Count','Sent_Count']].describe())


# Analysis for Spam Messages 
print()
print("Analysis for Spam Messages ")
print(df[df['Target']==1][['num_characters','Word_Count','Sent_Count']].describe())


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
plt.show()


# Pair Plot
sns.pairplot(df,hue='Target')
plt.show()

# Select only numeric columns for correlation 

numeric_df=df[['Target','num_characters','Word_Count','Sent_Count']]
plt.figure(figsize=(10,5))
sns.heatmap(numeric_df.corr(),annot=True,cmap='PiYG')
# bwr,sesimic,PiYG,PrGn,RdBu
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()