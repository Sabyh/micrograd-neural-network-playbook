import numpy as np

# Step 1: Tiny Tensor class with autograd (from before)
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)  # Storing the actual value (like a number)
        self.requires_grad = requires_grad       # Tells if we need to track this for learning
        self.grad = 0.0                          # Gradient, which is how much we change the value by
        self._backward = lambda: None            # Function for going backward and calculating gradients
        self._prev = []                          # Stores previous operations
        self._op = ""                            # Stores the operation (like '+' or '*')

    def __add__(self, other):
        out = Tensor(self.data + other.data, requires_grad=True)
        out._prev = [self, other]

        def _backward():
            if self.requires_grad:
                self.grad += 1.0 * out.grad
            if other.requires_grad:
                other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, requires_grad=True)
        out._prev = [self, other]

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0          # Start from the last node (output)
        self._backward()         # Call the backward function to calculate gradients
        for prev in self._prev:
            prev._backward()     # Go back to previous nodes and calculate their gradients

# Step 2: Training our neuron
x = Tensor(2.0, requires_grad=False)     # input
y_true = Tensor(6.0, requires_grad=False) # target output

# We want to learn these
w = Tensor(1.0, requires_grad=True)  # guess weight
b = Tensor(0.0, requires_grad=True)  # guess bias

# Step 3: Training loop
for epoch in range(20):
    # Forward pass: y = w * x + b
    y_pred = w * x + b

    # Loss: (y_pred - y_true)^2
    error = y_pred + Tensor(-y_true.data)   # subtract
    loss = error * error

    # Reset grads
    w.grad = 0
    b.grad = 0

    # Backward pass
    loss.backward()

    # Update weights
    lr = 0.1
    w.data -= lr * w.grad
    b.data -= lr * b.grad

    print(f"Epoch {epoch}: Loss = {loss.data:.4f}, w = {w.data:.4f}, b = {b.data:.4f}")
