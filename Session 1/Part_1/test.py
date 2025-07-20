import numpy as np

print()
print("Single Dimensional Array ")
n1=np.array([10,20,30,40,50])
print(n1)

print()
print("Multi Dimensional Array ")
n1=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(n1)

print()
print("Multi Dimensional Array using Arrange func")
n1=np.arange(10,20)
print(n1)

print()
print ("Multi Dimensional Array filed with Zeros ")
n1=np.zeros([5,5])
print(n1)

print()
print("Single Dimensional Array Using Radom Randint")
n1=np.random.randint(1,100,5)
print(n1)

print()
print("Multi Dimensional Arrays Horizontal Stacked")
n1=np.array([10,20,30,40,50])
n2=np.array([1,2,3,4,5])
print(np.hstack((n1,n2)))

print()
print("Multi Dimensional Arrays Vertical Stacked")
n1=np.array([10,20,30,40,50])
n2=np.array([1,2,3,4,5])
print(np.vstack((n1,n2)))

print()
print("Multi Dimensional Arrays Column Stacked")
n1=np.array([10,20,30,40,50])
n2=np.array([1,2,3,4,5])
print(np.column_stack((n1,n2)))

print()
print("Intersection of Both Arrays")
n1=np.array([10,20,30,40,50])
n2=np.array([60,70,80,40,50])
print(np.intersect1d(n1,n2))

print()
print("Difference of Both Arrays")
n1=np.array([10,20,30,40,50])
n2=np.array([60,70,80,40,50])
print("n1-n2")
print(np.setdiff1d(n1,n2))
print("n2-n1")
print(np.setdiff1d(n2,n1))

print()
print("Sum of Two Arrays")
n1=np.array([10,20,30,40,50])
n2=np.array([1,2,3,4,5])
print(np.sum([n1,n2]))

print()
print("Sum of Two Arrays on axis")
n1=np.array([10,20,30,40,50])
n2=np.array([1,2,3,4,5])
print("Axis = 0")
print(np.sum([n1,n2],axis=0))
print(np.sum([n2,n1],axis=0))
print("Axis = 1")
print(np.sum([n1,n2],axis=1))
print(np.sum([n2,n1],axis=1))


print()
print("Arithematic Ops on Array")
n1=np.array([10,20,30,40,50])
n2=n1+2
n3=n1-4
n4=n1*7
n5=n1**2
n6=n1/2
n7=n1%3
print(f"Array = n1 | Original : {n1}")
print(f"Array = n2 | Ops Addition : {n2}")
print(f"Array = n3 | Ops Subtraction : {n3}")
print(f"Array = n4 | Ops Multiplication : {n4}")
print(f"Array = n5 | Ops Power : {n5}")
print(f"Array = n6 | Ops Division : {n6}")
print(f"Array = n7 | Ops Modulo : {n7}")

print()
print("Mean of an Array")
n1=np.array([10,20,30,40,50])
print(np.mean(n1))

print()
print("Median of an Array")
n1=np.array([10,20,30,40,50,60])
print(np.median(n1))

print()
print("Standard Deviation of an Array")
n1=np.array([10,20,30,40,50])
print(n1)
print(np.std(n1))
n1=np.array([10,20,30,40,50,60])
print(n1)
print(np.std(n1))