# Tiny AutoDiff
A tiny package for computing the gradients of vector to scalar functions using [Reverse Mode Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)
## Working
- Gradient computation of most basic functions
- Full computational graph functionality 
- Overloaded operators

## Examples
Full list of examples are given in `autodiff_tests.py`. Here are a few:
- Computing derivative of the function, <img src="https://render.githubusercontent.com/render/math?math=f(x_1,x_2) = \ln(x_1) %2B x_1 x_2 - \sin(x_2)"> at <img src="https://render.githubusercontent.com/render/math?math=(x_1,x_2) = (2,5)">

```
from Node import *
from Computational_Graph import *
import Grad_ops as ops

#Initialize computational graph
graph = Computational_Graph()

def f(x1, x2):
    return ops.ln(x1) + x1*x2 - ops.sin(x2)

#Initialize leaf (variable) nodes
a = Node(2, 'Leaf', graph)
b = Node(5, 'Leaf', graph)

# Adding values to the computational graph
graph(a)
graph(b)
print(f(a,b).value)
11.652
grad = graph.backward()
print(grad)
[5.5, 1.716]
```

- Computing derivative of the function, <img src="https://render.githubusercontent.com/render/math?math=f(x_1,x_2) = x_1 x_2 - e^{x_1 - x_2}\sin(x_1)"> at <img src="https://render.githubusercontent.com/render/math?math=(x_1,x_2) = (3,2)">
```
from Node import *
from Computational_Graph import *
import Grad_ops as ops

#Initialize computational graph
graph = Computational_Graph()

def g(x1, x2):
        return x1*x2 - ops.exp(x1-x2)*ops.sin(x1)

#Initialize leaf (variable) nodes
a = Node(3, 'Leaf', graph)
b = Node(2, 'Leaf', graph)

# Adding values to the computational graph
graph(a)
graph(b)
print(g(a,b).value)
5.616
grad = graph.backward()
print(grad)
[4.307, 3.383]
```
## To Do:
- Improve how the graph and values interface

