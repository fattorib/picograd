import numpy as np
from Node import *
import Computational_Graph as G


class Tensor():
    def __init__(self, values, graph, *args):
        self.arr = [Variable(i, graph) for i in values]
        self.graph = graph
        self.item = values
        self.len = len(values)

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
    a = Tensor([0, 2], graph)
    # a = Variable(0, graph)
    # b = Tensor([4, 5], graph)

    # print(dot(a, b).value)
    print(sum(a))
    grad = graph.backward()
    print(grad)
