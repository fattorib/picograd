import numpy as np
from Node import *
import Computational_Graph as G


def sin(a):
    graph = a.graph

    v = np.sin(a.value)

    # Create new node
    v_i = Node(v, 'Sine', graph, False)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def cos(a):
    graph = a.graph

    v = np.cos(a.value)

    # Create new node
    v_i = Node(v, 'Cosine', graph, False)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def ln(a):
    graph = a.graph

    v = np.log(a.value)

    # Create new node
    v_i = Node(v, 'Natural Logarithm', graph, False)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def exp(a):
    graph = a.graph
    v = np.exp(a.value)

    # Create new node
    v_i = Node(v, 'Exponential', graph, False)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


if __name__ == "__main__":

    print('Hey')
