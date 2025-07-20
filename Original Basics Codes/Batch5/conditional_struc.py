print()
print("|   INPUT OF DATA   |")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
c = int(input("Enter a Number (c) :"))
print(a)
print(b)
print(c)
print()
print("|   SIMPLE IF STATEMENT   |")
print()
if(a>b):
    print(f"{a} is greater than {b}")

if(a<b):
    print(f"{a} is lesser than {b}")

print()
print("|   IF ELSE STATEMENT   |")
print()
if(a>b):
    print(f"{a} is greater than {b}")
else:
    print(f"{b} is greater than {a}")

print()
print("|   IF ELSE LADDER STATEMENT   |")
print()
if(a==0):
    print(f"Value of a is 0")
elif(b==0):
    print(f"Value of b is 0")
elif(c==0):
    print(f"Value of c is 0")
else:
    print(f" All the values are Greater than 0")
    
print()
print("|   FOR LOOP   |")
print()

List_l=[1,2,3,4,5,"apple","banana","peach","Watermelon","Mango"]
for i in List_l:
    print(i)
    
str="DRAGONFRUIT"
for x in str:
    print(x)

sum=0
print("Sum of number from 0 to 6")
for j in range(6):
    print(j)
    sum=sum+j
print(f"Sum of numbers from 0 to 6 is {sum}")

print()
print("|   WHILE LOOP   |")
print()

k=0
while(k<=5):
    print(f"Value of k is {k}")
    k=k+1