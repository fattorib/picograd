import numpy as np
from Node import *
import Computational_Graph as G


class Tensor():
    def __init__(self, values, graph, *args):
        self.arr = [Variable(i, graph) for i in values]
        self.graph = graph
        self.item = values
        self.len = len(values)
        for var in self.arr:
            self.graph(var)

    def __len__(self):
        return self.len


def sum(x):
    # Return elementwise sum of tensor x
    c = 0
    for element in x.arr:
        c += element
    return c


def Hadamard(x, y):
    # Get graph
    graph = x.graph
    had_arr = []
    for i, j in zip(x.arr, y.arr):
        had_arr.append((i*j).value)
    return Tensor(had_arr, graph)


def dot(x, y):
    assert len(x) == len(y), "Check your input tensors, lengths do not match"
    return (sum(Hadamard(x, y)))


if __name__ == "__main__":
    graph = G.Computational_Graph()
    a = Tensor([1, 2, 3], graph)
    # b = Tensor([-1, 0, 2], graph)

    print(graph.graph_visualize_list())
