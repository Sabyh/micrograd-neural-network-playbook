# Youtube videos we are following 

https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ


https://www.youtube.com/watch?v=kCc8FmEb1nY&t=608s


https://www.youtube.com/watch?v=1aM1KYvl4Dw


https://www.youtube.com/watch?v=5avSMc79V-w


https://www.youtube.com/watch?v=RFaFmkCEGEs


https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh


### how does gpu works

https://www.youtube.com/watch?v=OUzm06YaUsI


### RNN

https://www.youtube.com/watch?v=AsNTP8Kwu80&list=PLjYRg5G2xJSnccwkbKQOI6AS2yITSllow


----------------------------------------------------------------------------

https://www.youtube.com/watch?v=Rs_rAxEsAvI


https://www.datacamp.com/tutorial/minimax-algorithm-for-ai-in-python


https://www.datacamp.com/tutorial/tutorial-monte-carlo


https://www.youtube.com/watch?v=mK_PfqM88OY&list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu&index=5


https://www.youtube.com/watch?v=uxeqipNwuP0


https://www.youtube.com/watch?v=jjckEViwt_o


https://www.youtube.com/watch?v=Ih5Mr93E-2c&list=PLn5tx65egy3c0LeTpRxz_iHsbOcYhNFHF


https://www.youtube.com/watch?v=9RN2Wr8xvro


https://www.youtube.com/watch?v=oJNHXPs0XDk


https://www.youtube.com/watch?v=ILsA4nyG7I0


https://www.youtube.com/watch?v=hfMk-kjRv4c


### deepseek good explanation

https://www.youtube.com/watch?v=jjckEViwt_o

### llm course

https://huggingface.co/learn/llm-course/chapter1/1


# micrograd-neural-network-playbook

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
