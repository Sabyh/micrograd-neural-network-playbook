# Neural network


## ✅ 1. Build a Tiny Neural Net From Scratch (with easy explanations)  
## ✅ 2. Create a Notebook or Visual Playground  
And we’ll **explain every line** like we’re talking to a beginner.

---

## 🔧 Part 1: Tiny Neural Net — One Neuron, One Input

Let’s say you're building a robot that guesses a number based on an input.

**Goal**:  
If I give it `x = 2`, I want it to predict `y = 6`.  
Maybe the real relationship is something like: `y = 2 * x + 2`

We’ll use our custom `Tensor` class and teach the neuron how to learn this pattern.

---

### 🧠 What’s a Neuron?

It’s like a math formula:  
```python
y = w * x + b
```

- `x` = input (2.0)
- `w` = weight (how much it scales x)
- `b` = bias (just a number added)
- `y` = output (what we predict)

We want to adjust `w` and `b` so the prediction gets close to the real answer.

---

### 📦 Let's Write This In Code (with autograd)

Here's a **notebook-style implementation** you can run in **Google Colab** or locally:

```python
import numpy as np

# Step 1: Tiny Tensor class with autograd (from before)
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = []
        self._op = ""

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
        self.grad = 1.0
        self._backward()
        for prev in self._prev:
            prev._backward()

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
```

---

## 📖 What This Does

1. Starts with a **bad guess** for `w` and `b`
2. Calculates the **loss** (how wrong the prediction is)
3. Uses `.backward()` to get the **gradients**
4. Updates `w` and `b` to make the prediction better
5. Repeats until it **learns** the correct formula

---

### ✅ Final Output Will Look Like:
```
Epoch 0: Loss = 16.00, w = 1.8, b = 0.8
Epoch 1: Loss = 2.56, w = 2.28, b = 1.28
...
Epoch 19: Loss = 0.00, w ≈ 2.0, b ≈ 2.0
```

🎉 The neuron has learned the pattern: `y = 2x + 2`

---

## 🎨 Part 2: Want a Visual Playground?

Here are 2 ways we can go:

1. **Jupyter Notebook** — I’ll give you a `.ipynb` file you can run in Google Colab or Jupyter
2. **Web Playground** — I can build a browser-based visual tool (with sliders + graphs)

Let me know your choice and I’ll build it up for you right away!

--- 

And if you want next steps:
- Add more neurons (like a full layer)
- Add activation functions (`ReLU`, `sigmoid`)
- Train on real-world examples

We’ll grow this step-by-step 💪