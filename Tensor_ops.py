import numpy as np
from Node import *
import Computational_Graph as G
from Tensor import *


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


if __name__ == "__main__":
    graph = G.Computational_Graph()

    a = Tensor([1, 2], graph, requires_grad=True)
    b = Tensor([3, 4], graph, requires_grad=True)

    c = dot(a, b)

    print(c.value)
    print(graph.backward())

    mat = np.array([[1, 2], [1, 1]])

    c = Tensor(mat, graph, requires_grad=True)
    print(c)
