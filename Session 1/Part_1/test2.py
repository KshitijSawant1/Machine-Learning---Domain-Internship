import pandas as pd
print()
print("Creating Basic Series int")
s1=pd.Series([1,2,3,4,5])
print(s1)

print()
print("Creating Basic Series float")
s1=pd.Series([1.0,2.0,3.0,4.0])
print(s1)

print()
print("Creating Basic Series obj")
s1=pd.Series(['a','b','c','d','e'])
print(s1)

print()
print("Creating Basic Series mixed")
s1=pd.Series([1,2.0,'c',5])
print(s1)

print()
print("Chaning Index")
s1=pd.Series([1,2,3,4,5],
             index=['a','b','c','d','e'])
print(s1)
print(s1['c'])

print()
print("Series using Dictionary")
s1=pd.Series({
    'a':10,'b':20,'c':30
})
print(s1)

print()
print("Series using Dictionary Multi Values")
s1=pd.Series({
'Name':['Alex','Ben','Clark'],
'Marks':[90,80,70]
})
print(s1)

print()
print("Slicing")
s1=pd.Series([1,2,3,4,5])
print(s1[1:3])
print()
s1=pd.Series([1,2,3,4,5],
             index=['a','b','c','d','e'])
print(s1['b':'d'])


print()
print("Negative Indexing")
s1=pd.Series([1,2,3,4,5])
print(s1)
print(f"Negative Index at -1: {s1.iloc[-1]}")
print(f"Negative Index at -3: {s1.iloc[-3]}")
print(f"Negative Index at -4: {s1.iloc[-4]}")


print()
print("Series using Dictionary Multi Values")
s1=pd.Series({
'Name':['Alex','Ben','Clark'],
'Marks':[90,80,70]
})
print(s1)

print()
print("Dataframe")
s1=pd.DataFrame({
    'Name':['Alex','Ben','Clark'],
    'Marks':[90,80,70]})
print(s1)