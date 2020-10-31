import numpy as np
from Node import *
import Grad_Array as G


def _add(a, b, graph):
    # a,b have to be nodes of our graph
    v = a.value+b.value

    # Create new node
    v_i = Node(v, 'addition')

    # Add it to existing graph
    graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_i)
    b.parents.append(v_i)

    return v_i


def _multiply(a, b, graph):

    v = a.value*b.value

    # Create new node
    v_i = Node(v, 'multiplication')

    # Add it to existing graph
    graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_i)
    b.parents.append(v_i)

    return v_i


def _subtract(a, b, graph):

    v = a.value-b.value

    # Create new node
    v_i = Node(v, 'subtraction')

    # Add it to existing graph
    graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_i)
    b.parents.append(v_i)

    return v_i


def _sin(a, graph):

    v = np.sin(a.value)

    # Create new node
    v_i = Node(v, 'sine')

    # Add it to existing graph
    graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_i)

    return v_i


def _ln(a, graph):

    v = np.log(a.value)

    # Create new node
    v_i = Node(v, 'ln')

    # Add it to existing graph
    graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_i)

    return v_i


if __name__ == "__main__":

    graph = G.Computational_Graph()

    # Clean up how this is done
    a = Node(2, 'leaf')
    b = Node(5, 'leaf')

    graph(a)
    graph(b)

    # Using function from https://arxiv.org/pdf/1502.05767.pdf for testing

    def f(x1, x2):
        return (_subtract(_add(_ln(x1, graph), _multiply(x1, x2, graph), graph), _sin(x2, graph), graph))

    print(f(a, b).value)
    graph.graph_visualize_list()
