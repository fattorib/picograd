import numpy as np
from Node import *
import Grad_Array as G


def sin(a):
    graph = a.graph

    v = np.sin(a.value)

    # Create new node
    v_i = Node(v, 'Sine', graph)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def cos(a):
    graph = a.graph

    v = np.cos(a.value)

    # Create new node
    v_i = Node(v, 'Cosine', graph)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def ln(a):
    graph = a.graph

    v = np.log(a.value)

    # Create new node
    v_i = Node(v, 'Natural Logarithm', graph)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


def exp(a):
    graph = a.graph
    v = np.exp(a.value)

    # Create new node
    v_i = Node(v, 'Exponential', graph)

    # Add it to existing graph
    v_idx = graph(v_i)

    # Tell a and b that v_i node is a parent
    a.parents.append(v_idx)

    return v_i


if __name__ == "__main__":

    graph = G.Computational_Graph()

    # Clean up how this is done
    a = Node(2, 'Leaf', graph)
    b = Node(5, 'Leaf', graph)

    # Adding to the computational graph
    graph(a)
    graph(b)

    # Using function from https://arxiv.org/pdf/1502.05767.pdf for testing
    def f(x1, x2):
        return ln(x1) + x1*x2 - sin(x2)

    print(f(a, b).value)
    # graph.graph_visualize_list()
    # print()
    grad = graph.backward()
    print(grad)

    graph = G.Computational_Graph()

    # Clean up how this is done
    a = Node(3, 'Leaf', graph)
    b = Node(2, 'Leaf', graph)

    graph(a)
    graph(b)

    def g(x1, x2):
        return x1*x2 - exp(x1-x2)*sin(x1)

    print(g(a, b).value)
    grad = graph.backward()
    print(grad)

    # # Operator Overloading Examples
    # graph = G.Computational_Graph()

    # # Clean up how this is done
    # a = Node(0, 'Leaf', graph)
    # b = Node(0, 'Leaf', graph)

    # graph(a)
    # graph(b)

    # def g(x1, x2):
    #     return _sin(x1) + _cos(x2)

    # print(g(a, b).value)
    # grad = graph.backward()
    # print(grad)
