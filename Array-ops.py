import numpy as np
from Node import *
import Computational_Graph as G


class Tensor():
    def __init__(self, values, graph, *args):
        self.arr = np.array([Variable(i, graph) for i in values])
        self.graph = graph
        self.item = values
        self.len = len(values)
        self.shape = self.arr.shape

    def __len__(self):
        return self.len


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


if __name__ == "__main__":
    graph = G.Computational_Graph()
    a = Tensor([1, 1, 3], graph)
    b = Tensor([2, 2, 0], graph)

    print(dot(a, b).value)
    grad = graph.backward()
    print(grad)
