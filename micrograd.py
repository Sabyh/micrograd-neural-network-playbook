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
      self._backward = lambda: None 
      self.label = label if label is not None else str(data)
        
    def __repr__(self):
      return f"Value({self.data})"
    
    def __add__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      out = Value(self.data + other.data, _children=(self, other), _op='+')
      def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
      out._backward = _backward
      return out

    def __mul__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      out = Value(self.data * other.data, _children=(self,other), _op='*')
      def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
      out._backward = _backward
      return out 
      
    def tanh(self):
      x = self.data
      t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
      out = Value(t, _children=(self,), _op='tanh')
      def _backward():
        self.grad += (1 - t**2) * out.grad
      out._backward = _backward
      return out
      
    def backward(self):
      topo = []
      visited = set()
      def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
      build_topo(self)
      self.grad = 1.0
      print("TOPPPPPPPPPOOOOOO")
      print(topo)
      for node in reversed(topo):
        node._backward()



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
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

e = a * b
d = e + c
L = d * f

print(L.data)


# Save the graph to see it
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


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o =  n.tanh()


o.grad = 1.0
# o.backward()
# n.backward()
# x1w1x2w2.backward()
# x1w1.backward()
# x2w2.backward()
# w1.backward()
# x1.backward()
# w2.backward()
# x2.backward()
# b.backward()


o.backward()

a = Value(3.0, label='a')

b = a + a; b.label = 'b'
b.backward()


dot = draw_dot(b)
try:
    dot.render('computational_graph', cleanup=True)
    print("Graph saved as computational_graph.svg")
except Exception as e:
    print(f"Could not save graph: {e}")
    print("Make sure Graphviz is installed: pip install graphviz")


