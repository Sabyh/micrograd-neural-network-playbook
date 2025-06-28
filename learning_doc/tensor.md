
# TensorA tensor is a mathematical object that generalizes scalars, vectors, and matrices to higher dimensions. In the context of machine learning and deep learning, tensors are used to represent data in a structured way.
Tensors can be thought of as multi-dimensional arrays, where the number of dimensions is referred to as the tensor's rank. For example:
- A scalar is a 0-dimensional tensor (a single number). 
- A vector is a 1-dimensional tensor (a list of numbers).
- A matrix is a 2-dimensional tensor (a table of numbers).
- A 3-dimensional tensor can be thought of as a cube of numbers, and so on.
- Tensors can have any number of dimensions, and the number of elements in a tensor is determined by its shape (the size of each dimension).
- Tensors are used in various machine learning frameworks, such as TensorFlow and PyTorch, to represent input data, model parameters, and outputs.
- Tensors can be manipulated using various mathematical operations, such as addition, multiplication, and reshaping.

# what is tensor
Tensor is nothing more than a multi-dimensional array with some custom operations that they do in a optimized way.
in optimization what they do is build a execution graph and then they do the optimization on that graph.
so basically all the operations are done on the graph instead of directly as we do in numpy or on a array as that operation might be costly in a large dataset

# tensor operations
- Tensor operations are mathematical operations that can be performed on tensors, which are multi-dimensional arrays.
- These operations include:
  - Element-wise operations: Operations that are applied to each element of the tensor independently, such as addition, subtraction, multiplication, and division.
  - Matrix operations: Operations that are applied to matrices (2-dimensional tensors), such as matrix multiplication, transpose, and inverse.
  - Reduction operations: Operations that reduce the dimensions of a tensor, such as summation, mean, and max.
  - Broadcasting: A technique that allows tensors of different shapes to be combined in a way that makes sense mathematically.
  - Reshaping: Changing the shape of a tensor without changing its data.
  - Slicing: Extracting a portion of a tensor by specifying indices or ranges.


# Graphs in these tensor libraries
we have to types of graph static and dynamic graph
- Static graph: A static graph is a graph that is defined before the computation begins. The structure of the graph does not change during the execution of the program. This means that all the operations and their dependencies are known in advance. Static graphs are typically used in frameworks like TensorFlow 1.x, where the computation graph is built first and then executed.
- Dynamic graph: A dynamic graph is a graph that is defined during the execution of the program. The structure of the graph can change as the program runs, allowing for more flexibility in defining computations. This means that operations can be added or modified on-the-fly. Dynamic graphs are typically used in frameworks like PyTorch, where the computation graph is built as operations are executed.
- The choice between static and dynamic graphs depends on the specific use case and the requirements of the application. Static graphs can be more efficient for certain types of computations, while dynamic graphs can be more flexible and easier to work with for certain types of models.
- TensorFlow uses a static computation graph, which means that the graph is defined before the execution of the operations. This allows for optimizations to be performed on the graph before it is executed.
- PyTorch uses a dynamic computation graph, which means that the graph is built on-the-fly as operations are executed. This allows for more flexibility in defining computations and makes it easier to work with models that have variable input sizes or structures.

## Example of a dynamic graph
```
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other), _op='add')
        
        def _backward():
            if self.requires_grad:
                self.grad = 1 * out.grad
            if other.requires_grad:
                other.grad = 1 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        out = Tensor(self.data * other.data, requires_grad=True, _children=(self, other), _op='mul')
        
        def _backward():
            if self.requires_grad:
                self.grad = other.data * out.grad
            if other.requires_grad:
                other.grad = self.data * out.grad
        out._backward = _backward
        return out
```

Dynamic graph: Just record each operation and its parents (like a linked list of ops).

Static graph: Build a full graph representation and execute later (more like a compiler).

# Example of a static graph
```class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None

    def __add__(self, other):
        return Tensor(self.data + other.data, requires_grad=True)

    def __mul__(self, other):
        return Tensor(self.data * other.data, requires_grad=True)
```


let's **build a mini tensor library** from scratch with:

- Tensors that support basic math: `+`, `*`, `**`, etc.
- **Autograd**: computes gradients with `.backward()`
- **Dynamic computation graph**: tracks operations on the fly

---

### ✅ STEP 1: Minimal Tensor Class with Autograd

We'll implement a `Tensor` class in pure Python + NumPy, similar to PyTorch or Tinygrad.

```python
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = 0.0  # Start with scalar grad for simplicity
        self._backward = lambda: None
        self._prev = set(_children)  # Parents in the graph
        self._op = _op  # Operation that created this tensor

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other), _op='add')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=True, _children=(self, other), _op='mul')

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data ** power, requires_grad=True, _children=(self,), _op='pow')

        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort to process parents before children
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        self.grad = 1.0  # seed gradient
        for t in reversed(topo):
            t._backward()
```

---

### ✅ STEP 2: Try It Out

```python
# Forward computation
x = Tensor(2.0, requires_grad=True)
y = x * x + x * 3 + 1  # y = x^2 + 3x + 1

# Backward pass
y.backward()

print("x:", x)
print("y:", y)
print("dy/dx (x.grad):", x.grad)
```

### 🧠 What's Happening?

Let's compute it by hand first:

```
y = x^2 + 3x + 1
dy/dx = 2x + 3 → at x = 2, dy/dx = 2*2 + 3 = 7
```

You should see:
```bash
x: Tensor(data=2.0, grad=7.0)
y: Tensor(data=11.0, grad=1.0)
dy/dx (x.grad): 7.0
```

---

### 🔁 What's Happening Internally?

- Each operation like `x * x` builds a new `Tensor`, and stores:
  - the inputs (`_children`)
  - the operation (`_op`)
  - how to compute the gradient (`_backward`)
- When `.backward()` is called on `y`, it:
  - builds a topological order (DAG traversal)
  - calls `_backward()` on each node in **reverse**
  - accumulates `.grad` values

---

## ✅ What's Missing in This Toy Example?

- Broadcasting, matrix ops, indexing, slicing
- Vector/matrix shape handling
- Gradient accumulation for non-scalars
- GPU support (you’d use `cupy` instead of `numpy`)
- More ops: `log`, `exp`, `tanh`, `relu`, etc.

---

## 🚀 Next Steps

Would you like me to:
1. Add support for **matrices / vector gradients**?
2. Show how to **plot the computation graph** like in PyTorch viz tools?
3. Implement this in a **Jupyter Notebook with visual debug output**?
4. Add **GPU support** using `cupy`?

Let me know what you'd like to explore next!