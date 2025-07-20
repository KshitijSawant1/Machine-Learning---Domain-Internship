import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print()
print("Line Graph")
# x = np.random.randint(1,100,10)
x = np.arange(1,50)
a=x+2
b=x-5
c=x/4
d=x%8
e=x*3
f=x*2
print(x)
plt.plot(x,a)
plt.plot(x,b)
plt.plot(x,c)
plt.plot(x,d)
plt.plot(x,e)
plt.plot(x,f)
plt.title("Line Plot Graph")
plt.show()