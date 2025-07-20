print()
print("|   INPUT OF DATA   |")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
print(f"Value of A : {a}")
print(f"Value of B : {b}")

print()
print("|   ARITHEMATIC OPERATOR   |")
print()
print(f"Addition of {a} and {b}       : {a+b}")
print(f"Subtraction of {a} and {b}    : {a-b}")
print(f"Multiplication of {a} and {b} : {a*b}")
print(f"Exponent of {a} and {b}       : {a**b}")
print(f"Division of {a} and {b}       : {a/b}")
print(f"Floor Division of {a} and {b} : {a//b}")
print(f"Modulus of {a} and {b}        : {a%b}")


print()
print("|   CONDITIONAL OPERATOR   |")
print()
if(a==b):
    print(f"Value of {a} is equal to {b} ")
if(a!=b):
    print(f"Value of {a} is not equal to {b} ")
if(a>=b):
    print(f"Value of {a} is greater than or equal to {b} ")
if(a<=b):
    print(f"Value of {a} is lesser than or eqaul to {b} ")
if(a>b):
    print(f"Value of {a} is greater than {b} ")
if(a<b):
    print(f"Value of {a} is lesser than  {b} ")

print()
print("|   LOGICAL OPERATOR   |")
print()

print(f"Value of {a} is logical and to {b} : {a and b}")
print(f"Value of {a} is logical or to {b}  : {a or b}")

bool_value=True
print(f"Logical not value of {bool_value} : {not bool_value}")