import math
import numpy as np
import matplotlib.pyplot as plt

#https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
#simple derivative function
#def derivative(f, x, h=1e-5):
#    return (f(x + h) - f(x)) / h

def f(x):
    return 3*x**2 - 4*x + 5
  
f(3.0)

xs = np.arange(-5, 5, 0.25)

print(xs)

ys = f(xs)
print(ys)

plt.plot(xs, ys)


h  = 0.001
x = 3.0
print(x+h)

