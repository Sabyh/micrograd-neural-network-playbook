import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

# Your original function code (unchanged)
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
value = f(x+ h)
print(value)
change_in_value = f(x + h) - f(x)
print(change_in_value)
derivative = change_in_value / h
print(derivative)

# Three variables derivative (unchanged)
h = 0.001
a = 3.0
b = -4.0
c =  1.0
d1 = a*b + c
a += h
d2 = a*b + c
print("d1:", d1)
print("d2:", d2)
print("h:", h)
print("d2 - d1:")   
print(d2 - d1)
derivative = (d2 - d1) / h
print("Derivative of the function at x = 3.0:")
print(derivative)

# Fixed Value class - only minimal changes
class Value:
    def __init__(self, data, _children=(), _op='', label=None):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0 
        self.label = label if label is not None else str(data)
        
    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        if isinstance(other, Value):
            # FIXED: consistent parameter naming, no asterisks
            return Value(self.data + other.data, _children=(self, other), _op='+')
        return Value(self.data + other, _children=(self,), _op='+')  # FIXED: single child for scalar
    
    def __mul__(self, other):
        if isinstance(other, Value):
            # FIXED: consistent parameter naming with underscores
            return Value(self.data * other.data, _children=(self, other), _op='*')
        return Value(self.data * other, _children=(self,), _op='*')  # FIXED: single child for scalar

# Your original test code (unchanged)
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(1.0, label='c')
print("Value of a using micro grad:")
print(a)
print("Value of b using micro grad:")
print(b)
print("Value of a + b using micro grad:")
print(a + b)
d  = a * b + c
print("Value of d using micro grad:")
print(d)
print("Value of d.prev using micro grad:")
print(d._prev)
print("Value of d.op using micro grad:")
print(d._op)

e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

# Save the graph to see it
try:
    dot.render('computational_graph', cleanup=True)
    print("Graph saved as computational_graph.svg")
except Exception as e:
    print(f"Could not save graph: {e}")
    print("Make sure Graphviz is installed: pip install graphviz")

def lol():
  
  h = 0.001
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'
  f = Value(-2.0, label='f')
  L = d * f; L.label = 'L'
  L1 = L.data
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  b.data += h
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'
  f = Value(-2.0, label='f')
  L = d * f; L.label = 'L'
  L2 = L.data
  
  print((L2 - L1)/h)
print("Derivative of L with respect to A \n")
print("As we bumped A by h, we can see the derivative is close to the value we calculated before.")  
lol()

L.grad = 1.0
# Test the fixed version
dot = draw_dot(L)
print("Graph created successfully!")



L = d * f
# what will be dl/dd?
# dl/dd = f.data
# basic derivative rule
# f(x+h) - f(x)/h
# as L is d x f
# so derivate formula will be
# f(d + h) * f - d * f / h
# df + hf - df / h
# df cancel out
# hf/h
# so f is the answer


