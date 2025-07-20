print()
print("|   INPUT OF DATA   |")
print()
a = int(input("Enter a Number (a) :"))
b = int(input("Enter a Number (b) :"))
c = int(input("Enter a Number (c) :"))
print(f"Value of A = {a}")
print(f"Value of B = {b}")
print(f"Value of C = {c}")

def Find_Avg(a,b,c):
    print()
    print("|  Inside Average Function  |")
    print()
    sum= a+b+c
    avg = sum/3
    return avg

def Find_Factorial(a):
    print()
    print("|  Inside Factorial Function  |")
    print()
    cur_num = 0
    nxt_num = 1
    fib_num = nxt_num
    i=1
    while(i<=a):
        print(fib_num)
        i = i+1
        cur_num = nxt_num
        nxt_num = fib_num
        fib_num = cur_num+nxt_num
        
def Find_Greatest(a,b,c):
    print()
    print("|  Inside Greatest Function  |")
    print()
    if(a>b):
        if(a>c):
            print("Value of A is Greatest Amongst All")
        else:
            if(c>b):
                print("Value of C is Greatest Amongst All")
    else:
        if(b>c):
            print("Value of B is Greatest Amongst All")    
        else:
            print("Value of C is Greatest Amongst All")
            
            
            
Average_Result = Find_Avg(a,b,c)
print(f"Average Of All The 3 Inputs is : {Average_Result}")

Find_Factorial(a)
Find_Greatest(a,b,c)