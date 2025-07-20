print()
print("|---------------INPUT OF DATA----------------|")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
c = int(input("Enter a Number (c) :"))
print(f"Value of A : {a}")
print(f"Value of B : {b}")
print(f"Value of C : {c}")

print()
print("|------------SIMPLE IF STATEMENT-------------|")
print()
if(a>b):
    print(f"Value of A : {a} is Greater than Value of B : {b}")
if(c>b):
    print(f"Value of C : {c} is Greater than Value of B : {b}")

print()
print("|-------------IF ELSE STATEMENT--------------|")
print()
if(a>0):
    print(f"Value of A : {a} is a Positive Value")
else:
    print(f"Value of A : {a} is a Negative Value")

print()
print("|----------IF ELIF LADDER STATEMENT----------|")
print()
if(a>b):
    print(f"Value of A : {a} is Greater than Value of B : {b}")
elif(a>c):
    print(f"Value of A : {a} is Greater than Value of C : {c}")
elif(b>a):
    print(f"Value of B : {b} is Greater than Value of A : {a}")
elif(b>c):
    print(f"Value of B : {b} is Greater than Value of C : {c}")
elif(c>a):
    print(f"Value of C : {c} is Greater than Value of A : {a}")
elif(c>b):
    print(f"Value of C : {c} is Greater than Value of B : {b}")
else:
    print("Invalid Occurance")

print()
print("|------------------FOR LOOP------------------|")
print()
print("|===> TYPE 1 : in range|---------------------|")
for i in range(6):
    print(f"Value of i is : {i}")

print()
print("===> TYPE 2 : in data structure|-------------|")

list_l= [1,2,3,4.5,5.6,6.7,"Apple","Orange","Berry"]
for i in list_l:
    print(f"Data present in i : {i}")
    
print()
print("===> TYPE 3 : in a string|-------------------|")
str="WATERMELON"
for i in str:
    print(f"Alphabet of i : {i}")

print()
print("|-----------------WHILE LOOP-----------------|")
print()
print("ALl the Even Numbers from 0 to 100")
i=0
while i<=100:
    print(i ,end=" ,")
    i=i+2