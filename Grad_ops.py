import numpy as np
from Node import *
import Grad_Array as G


def _add(a, b, graph):
    # a,b have to be nodes of our graph
    v = a.value+b.value

    # Create new node
    v_i = Node(v, 'Addition')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)
    b.parents.append(v_idx)

    return v_i


def _multiply(a, b, graph):

    v = a.value*b.value

    # Create new node
    v_i = Node(v, 'Multiplication')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)
    b.parents.append(v_idx)

    return v_i


def _negative(a, graph):

    v = -1*a.value

    # Create new node
    v_i = Node(v, 'Negative')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def _recip(a, graph):

    # Add check to ensure no division by 0
    v = 1/a.value

    # Create new node
    v_i = Node(v, 'Reciprocal')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


# Compound operations
def _subtract(a, b, graph):
    return (_add(a, _negative(b, graph), graph))


def _divide(a, b, graph):
    return (_multiply(a, _recip(b, graph), graph))


def _sin(a, graph):

    v = np.sin(a.value)

    # Create new node
    v_i = Node(v, 'Sine')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def _cos(a, graph):

    v = np.cos(a.value)

    # Create new node
    v_i = Node(v, 'Cosine')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def _ln(a, graph):

    v = np.log(a.value)

    # Create new node
    v_i = Node(v, 'Natural Logarithm')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def _exp(a, graph):

    v = np.exp(a.value)

    # Create new node
    v_i = Node(v, 'Exponential')

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


if __name__ == "__main__":

    graph = G.Computational_Graph()

    # Clean up how this is done
    a = Node(2, 'Leaf')
    b = Node(5, 'Leaf')

    graph(a)
    graph(b)

    # Using function from https://arxiv.org/pdf/1502.05767.pdf for testing

    def f(x1, x2):
        return (_subtract(_add(_ln(x1, graph), _multiply(x1, x2, graph), graph), _sin(x2, graph), graph))

    print(f(a, b).value)
    # graph.graph_visualize_list()
    # print()
    graph.backward()

    # graph = G.Computational_Graph()

    # # Clean up how this is done
    # a = Node(2, 'Leaf')
    # b = Node(3, 'Leaf')

    # graph(a)
    # graph(b)

    # def g(x1, x2):
    #     return (_subtract(_multiply(x1, x2, graph), _sin(x2, graph), graph))
    # print(g(a, b).value)
    # graph.graph_visualize_list()
    # print()

    # def h(x1, x2):
    #     return _add(_multiply(x1, x2, graph), _sin(x1, graph), graph)

    # graph = G.Computational_Graph()

    # # Clean up how this is done
    # a = Node(2, 'Leaf')
    # b = Node(3, 'Leaf')

    # graph(a)
    # graph(b)

    # print('Value:', h(a, b).value)

    # graph.backward()
    # print()
