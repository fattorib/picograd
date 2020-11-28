import numpy as np
from Node import *
import Computational_Graph as G
from Array_ops import *


def sin(a):

    # Tensor support

    if type(a) == Node or type(a) == Variable:
        graph = a.graph

        v = np.sin(a.value)

        # Create new node
        v_i = Node(v, 'Sine', graph, False)

        # Add it to existing graph
        v_idx = graph(v_i)

        # Tell a and b that v_i node is a parent
        a.parents.append(v_idx)

        return v_i

    elif type(a) == Tensor:
        node_arr = []
        for i in a.arr:
            i = sin(i)
            node_arr.append(i)

        return Tensor(node_arr, a.graph, False)


def cos(a):
    if type(a) == Node or type(a) == Variable:
        graph = a.graph

        v = np.cos(a.value)

        # Create new node
        v_i = Node(v, 'Cosine', graph, False)

        # Add it to existing graph
        v_idx = graph(v_i)

        # Tell a and b that v_i node is a parent
        a.parents.append(v_idx)

        return v_i

    elif type(a) == Tensor:
        node_arr = []
        for i in a.arr:
            i = cos(i)
            node_arr.append(i)

        return Tensor(node_arr, a.graph, False)


def ln(a):

    if type(a) == Node or type(a) == Variable:
        graph = a.graph

        v = np.log(a.value)

        # Create new node
        v_i = Node(v, 'Natural Logarithm', graph, False)

        # Add it to existing graph
        v_idx = graph(v_i)

        # Tell a and b that v_i node is a parent
        a.parents.append(v_idx)

        return v_i

    elif type(a) == Tensor:
        node_arr = []
        for i in a.arr:
            i = ln(i)
            node_arr.append(i)

        return Tensor(node_arr, a.graph, False)


def exp(a):

    if type(a) == Node or type(a) == Variable:
        graph = a.graph
        v = np.exp(a.value)

        # Create new node
        v_i = Node(v, 'Exponential', graph, False)

        # Add it to existing graph
        v_idx = graph(v_i)

        # Tell a and b that v_i node is a parent
        a.parents.append(v_idx)

        return v_i

    elif type(a) == Tensor:
        node_arr = []
        for i in a.arr:
            i = exp(i)
            node_arr.append(i)

        return Tensor(node_arr, a.graph, False)


if __name__ == "__main__":
    graph = G.Computational_Graph()
    a = Tensor([1, 1, 1], graph, requires_grad=True)
    b = exp(a)
    print(b.value)
    print(graph.backward())
