import pandas as pd 
from sklearn.datasets import fetch_california_housing

data =fetch_california_housing()
df = pd.DataFrame(data.data,columns=data.feature_names)
df['Target'] = data.target
print("Data Loaded Successfully")

print()
print("Head of the Dataset Deafult : 5")
print()
print(df.head(10))

print()
print("Tail of the Dataset Deafult : 5")
print()
print(df.tail(10))

print()
print("Sample of the Dataset Deafult : 1")
print()
print(df.sample(10))


print()
print("Shape of the Dataset")
print()
print(df.shape)

print()
print("Summarize the Dataset")
print("Info")
print(df.info())
print()
print("Summary")
print(df.describe())


print()
print("Mimimum Values")
print()
print(df.min())

print()
print("Max Values")
print()
print(df.max())

print()
print("Count Values")
print()
print(df.count())

print()
print("Mean Values")
print()
print(df.mean())

print()
print("Median Values")
print()
print(df.median())

print()
print("Varience Values")
print()
print(df.var())

print()
print("Unique Value Counts")
print()
print(df.nunique())
print()

varcheck= df.nunique()
print(varcheck["MedInc"])

from sklearn.datasets import load_iris
data=load_iris()
print("Loaded Dataset Successfully")
df =pd.DataFrame(data.data,columns=data.feature_names)
df['Target']=data.target
print(df.head())
print(df.std())

# do Head, Tail , Sample , Min , Max , Count , 
# Std ,MEan , Medain , indo , shape , describe ,nunique