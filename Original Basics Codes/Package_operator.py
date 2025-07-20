from My_package.add_sub import add
from My_package.add_sub import sub
from My_package.mul_div import mul
from My_package.mul_div import div


print()
print("|---------------INPUT OF DATA----------------|")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
c = int(input("Enter a Number (c) :"))
d = int(input("Enter a Number (d) :"))
print(f"Value of A : {a}")
print(f"Value of B : {b}")
print(f"Value of C : {c}")
print(f"Value of D : {d}")
print()
print("|--------------ADDITION RESULT---------------|")
print()
print(f"Result of Addition of {a} and {b}    :{add(a,b)}")
print()
print("|-------------SUBTRACTION RESULT-------------|")
print()
print(f"Result of Subtraction of {a} and {b} :{sub(a,b)}")
print()
print("|-----------MULTIPLICATION RESULT------------|")
print()
print(f"Result of Multiplication of {a} and {b}    :{mul(c,d)}")
print()
print("|--------------DIVISION RESULT---------------|")
print()
print(f"Result of Division of {a} and {b} :{div(c,d)}")