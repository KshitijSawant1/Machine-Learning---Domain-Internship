print()
print("|   INPUT OF DATA   |")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
print(a)
print(b)
print()
print("|   ARITHEMATIC OPERATOR   |")
print()
print(f" Additon of {a} and {b} = {a+b}")
print(f"Subtraction of {a} and {b} = {a-b}")
print(f"Multiplication of {a} and {b} = {a*b}")
print(f"Divion of {a} and {b} = {a/b}")
print(f"Modulus of {a} and {b} = {a%b}")
print(f"Floor Division of {a} and {b} = {a//b}")
print(f"Exponent {a} and {b} = {a**b}")
print()
print("|   RELATIONAL OPERATOR   |")
print()
if(a==b):
    print(f"Value of {a} == {b}")
if(a!=b):
    print(f"Value of {a} != {b}")
if(a>b):
    {print(f"Value of {a} > {b} is greater than")}
if(a<b):
    {print(f"Value of {a} < {b} is less than")}
if(a>=b):
    {print(f"Value of {a} >= {b} is greater than or equal to")}
if(a==b):
    {print(f"Value of {a} <= {b} is lesser than or eqaul to ")}
print()
print("|   LOGICAL OPERATOR   |")
print()
if(a and b):
    print(f"Value of {a} and {b} if logical and operator valid")
if(a and b):
    print(f"Value of {a} or {b} if logical or operator valid")
c= True
print(f"c : {c}")
print(f"Value of {c} for logical not operator is {not c}")
print(f"Value of {a} for logical not operator is {not a}")
print(f"Value of {b} for logical not operator is {not b}")