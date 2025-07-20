print()
print("|---------------INPUT OF DATA----------------|")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
print(f"Value of A : {a}")
print(f"Value of B : {b}")

def add(x,y):
    return x+y

def sub(x,y):
    return x-y

print()
print("|--------------ADDITION RESULT---------------|")
print()
add_Result=add(a,b)
print(f"Result of Addition of {a} and {b}    :{add_Result}")
print()
print("|-------------SUBTRACTION RESULT-------------|")
print()
sub_Result=sub(a,b)
print(f"Result of Subtraction of {a} and {b} :{sub_Result}")