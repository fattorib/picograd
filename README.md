# Tiny AutoDiff
A tiny autodiff package for vector to scalar functions.
## Working
- Gradient computation of basic functions (add,subtract,multiply,divide, transcendentals)
- Computational graph 

## Examples
- Computing derivative of the function, <img src="https://render.githubusercontent.com/render/math?math=f(x_1,x_2) = \ln(x_1) %2B x_1 x_2 - \sin(x_1)"> at <img src="https://render.githubusercontent.com/render/math?math=(x_1,x_2) = (2,5)">

```
from Node import *
import Grad_Array as G
from Grad_ops import *

#Initialize computational graph
graph = G.Computational_Graph()

def f(x1, x2):
    return (_subtract(_add(_ln(x1, graph), _multiply(x1, x2, graph), graph), _sin(x2, graph), graph))

a = Node(2, 'Leaf')
b = Node(5, 'Leaf')

# Adding to the computational graph
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
import Grad_Array as G
from Grad_ops import *

#Initialize computational graph
graph = G.Computational_Graph()

a = Node(3, 'Leaf')
b = Node(2, 'Leaf')

graph(a)
graph(b)

def g(x1, x2):
        return (_subtract(_multiply(x1, x2, graph), _multiply(_exp(_subtract(x1, x2, graph), graph), _sin(x1, graph), graph), graph))
print(g(a,b).value)
5.616
grad = graph.backward()
print(grad)
[4.307, 3.383]
```
## To Do:
- Operator/function overloading to improve overall interface
- Improve how the graph and values interface
- Implement better tracking so "graph" doesn't need to be called for every operation
- Add power as an operation

