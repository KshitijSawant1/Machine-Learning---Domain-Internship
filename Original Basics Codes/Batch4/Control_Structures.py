print()
print("|-----------INPUT DATA------------|")
print()
a=int(input("Enter The Value Of A : "))
b=int(input("Enter The Value Of B : "))
c=int(input("Enter The Value Of C : "))
print("|---------------------------------|")
print(f"Value of A is    : {a}")
print(f"Value of B is    : {b}")
print(f"Value of C is    : {c}")
print("|---------------------------------|")

print()
print("|-------SIMPLE IF STATMENT--------|")
print()
if(a>0):
    print(f"Value of A is : {a} and is Positive")
if(b>10):
    print(f"Value of B is : {a} and is Greater than 10")
print("|---------------------------------|")

print()
print("|--------IF ELSE STATMENT---------|")
print()
if(c>15):
    print(f"Value of C is : {c} and is Greater than 15")
else:
    print(f"Value of C is : {c} and is Lesser than 15")
print("|---------------------------------|")

print()
print("|------IF ELIF ELSE STATMENT------|")
print()
if(a<0):
    print(f"Value of {a} is Lesser than 0")
elif(a%2==0):
    print(f"Value of {a} is Greater than 0 and EVEN")
else:
    print(f"Value of {a} is Greater than 0 and ODD")
print("|---------------------------------|")

print()
print("|-------NESTED IF STATMENT--------|")
print()
if(a>b):
    if(a>c):
        print(f"Value of A : {a} is the Greatest Amongst All The Others")
    elif(c>b):
        print(f"Value of C : {c} is the Greatest Amongst All The Others")
    else:
        print(f"Value of B : {b} is the Greatest Amongst All The Others")
else:
    if(b>c):
        print(f"Value of B : {b} is the Greatest Amongst All The Others")
    elif(c>a):
        print(f"Value of C : {c} is the Greatest Amongst All The Others") 
    else:
        print(f"Value of A : {a} is the Greatest Amongst All The Others")
        
print("|---------------------------------|")

print()
print("|-------------For Loop------------|")
print()
print("| Type 1 : IN RANGE---------------|")
print()
for i in range(6):
    print(f"Value of i is : {i}")
print()
print("| Type 2 : IN DATA STRUCTURE------|")
print()
List_L=[1,2,3,4.5,5.6,6.7,"Apple","Orange"]
for i in List_L:
    print(f"Value of i is : {i}")
print()
print("| Type 3 : IN STRING--------------|")
print()
str="WATERMELON"
for i in str:
        print(f"Alphabet of i is : {i}")
print()
print("|---------------------------------|")

print()
print("|------------While Loop-----------|")
print()
i=0
while(i<=5):
    print(f"Value of i is : {i}")
    i=i+1
print()
print("|---------------------------------|")