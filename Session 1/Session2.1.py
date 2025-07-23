import pandas as pd
import numpy as np 

#num=10
#for i in range(num):
#    n1=np.arange(i*10+1,i*10+6)
#    print(n1)
    
    
df = pd.read_csv("Session 1/Sample.csv")
print(df.head())
print()
print(df.tail())
print(df.describe())
print(df.info())
print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(),
            cbar=False,cmap='viridis'
            ,yticklabels=False)
plt.title("missing values")
plt.show()


# Sample Data
cat=['a','b','c','d','e']
val=[10,20,30,40,50]
plt.bar(cat,val,color='blue')
plt.title("Simple Bar Graph")
plt.xlabel("Cats")
plt.ylabel("Vals")
plt.show()


# Simple Line Graph
x=[10,20,30,40,50]
y=[1,2,3,4,5]

plt.plot(x,y,marker='*',linestyle='dashed'
         ,color="green")
plt.title("Simple Line Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Grouped Bar Graph
cat=['a','b','c','d','e']
val1=[10,20,30,40,50]
val2=[15,25,35,45,55]
x=np.arange(len(cat))
width=0.35
plt.bar(x-width/2,val1,width,color='skyblue')
plt.bar(x+width/2,val2,width,color='lightgreen')
plt.title("Grouped Bar Graph")
plt.xlabel("Cats")
plt.ylabel("Vals")
plt.show()

# Sample Data
cat=['a','b','c','d','e']
val=[10,20,30,40,50]
plt.barh(cat,val,color='blue')
plt.title("Horizontal Bar Graph")
plt.xlabel("Cats")
plt.ylabel("Vals")
plt.show()

# Stacked Bar Graph
cat=['a','b','c','d','e']
val1=[10,20,30,40,50]
val2=[15,25,35,45,55]
x=np.arange(len(cat))
width=0.35
plt.bar(cat,val2,width,color='skyblue')
plt.bar(cat,val1,width,color='lightgreen')
plt.title("Stacked Bar Graph")
plt.xlabel("Cats")
plt.ylabel("Vals")
plt.show()

#Pie chart 
cat=['a','b','c','d','e']
prec=[10,10,20,10,50]
plt.pie(prec,labels=cat,autopct='%1.1f%%',startangle=90)
plt.show()
