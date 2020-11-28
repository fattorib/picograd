import numpy as np
from Node import *
import Computational_Graph as G
from Tensor import *
from Grad_ops import ln


def sum(x):
    # Return elementwise sum of tensor x
    c = 0
    for element in x.arr:
        c += element
    return c


def dot(x, y):
    assert len(x) == len(y), "Check your input tensors, lengths do not match"
    d = 0
    for i, j in zip(x.arr, y.arr):
        d += i*j
    return d


def tensor_mm(x, y):
    # Perform matrix multiplication. LOL
    return None


def log(x):
    nodes_arr = []
    for i in x.arr:
        nodes_arr.append(ln(i))
    return Tensor(nodes_arr, graph, False)


if __name__ == "__main__":
    graph = G.Computational_Graph()

    a = Tensor([1, 2], graph, requires_grad=True)
    b = Tensor([1, 1], graph, requires_grad=True)
    print([a.arr[i].grad for i in range(0, len(a.arr))])

    def f(x, y):
        return dot(x, y)

    c = dot(a, b)
    print(c)
    grad = graph.backward()

    print(a.grad())
