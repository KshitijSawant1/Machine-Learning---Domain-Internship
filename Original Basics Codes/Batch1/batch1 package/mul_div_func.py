print()
print("|---------------INPUT OF DATA----------------|")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
print(f"Value of A : {a}")
print(f"Value of B : {b}")

def mul(x,y):
    return x*y

def div(x,y):
    return x/y

print()
print("|-----------MULTIPLICATION RESULT------------|")
print()
mul_Result=mul(a,b)
print(f"Result of Multiplication of {a} and {b}    :{mul_Result}")
print()
print("|--------------DIVISION RESULT---------------|")
print()
div_Result=div(a,b)
print(f"Result of Division of {a} and {b} :{div_Result}")