import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def Datacleaning(): 
    df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)
    df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['Target'] = encoder.fit_transform(df['Target'])
    print(df.isnull().sum())
    print(df.duplicated().sum())
    df.drop_duplicates(keep="first")
    print(df.shape)

# EDA
df = pd.read_csv("ProjectFile/spam new.csv", encoding='latin-1')
Datacleaning()

#Data Reading Target
print("\nData Reading Target")
print(df["Target"].value_counts())


# Pie chart
plt.pie(df["Target"].value_counts(), labels=["ham", "spam"], autopct="%0.2f%%", colors=["skyblue", "lightcoral"])
plt.title("Distribution of Ham vs Spam Messages")
plt.axis('equal')  
plt.show()

#pip install nltk
import nltk
# do it once then good nltk.download('punkt_tab')

# Character count
df['num_characters'] = df['Text'].apply(len)

# Word tokenization
df['Word_list'] = df['Text'].apply(lambda x: nltk.word_tokenize(x))

# Word count
df['Word_count'] = df['Word_list'].apply(len)

# Sentence tokenization
df['Sent_list'] = df['Text'].apply(lambda x: nltk.sent_tokenize(x))

# Sentence count
df['Sent_count'] = df['Sent_list'].apply(len)

# Quick preview
#Head and Describe
print(df[['Text', 'num_characters', 'Word_list', 'Word_count', 'Sent_list', 'Sent_count']].describe())

# Analysis for Ham Messages (Target = 0)
print("\nHam Message Statistics:")
print(df[df['Target'] == 0][['num_characters', 'Word_count', 'Sent_count']].describe())

# Analysis for Spam Messages (Target = 1)
print("\nSpam Message Statistics:")
print(df[df['Target'] == 1][['num_characters', 'Word_count', 'Sent_count']].describe())

# Set style
sns.set(style="whitegrid")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Histogram for Character Count
sns.histplot(data=df[df['Target'] == 0], x='num_characters', bins=30, color='lightgrey', label='Ham', kde=True, ax=axes[0])
sns.histplot(data=df[df['Target'] == 1], x='num_characters', bins=30, color='red', label='Spam', kde=True, ax=axes[0])
axes[0].set_title("Character Count Distribution")
axes[0].set_xlabel("Number of Characters")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# 2. Histogram for Word Count
sns.histplot(data=df[df['Target'] == 0], x='Word_count', bins=30, color='lightblue', label='Ham', kde=True, ax=axes[1])
sns.histplot(data=df[df['Target'] == 1], x='Word_count', bins=30, color='orange', label='Spam', kde=True, ax=axes[1])
axes[1].set_title("Word Count Distribution")
axes[1].set_xlabel("Number of Words")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.show()


sns.pairplot(df, hue='Target')
plt.show()

# Select only numeric columns for correlation
numeric_df = df[['Target','num_characters', 'Word_count', 'Sent_count']]
plt.figure(figsize=(10, 4))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
# coolwarm,bwr (blu,whit,red),sesimic (red,whit,blu),PiYG,PrGn,RdBu
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()



