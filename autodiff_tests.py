import numpy as np
from Node import *
import Computational_Graph as G
import Grad_ops as ops


"""
Single Variable Function Tests
"""


"""
Two Variable Function Tests
"""


def function_test(func, values, expected_outputs):
    # Init computational graph
    graph = G.Computational_Graph()
    values_array = []
    for i in values:
        k = Node(i, 'Leaf', graph)
        values_array.append(k)
        graph(k)

    out = func(values_array[0], values_array[1]).value
    grad = graph.backward()
    if out == expected_outputs[0] and grad == expected_outputs[1]:
        print('Test passed')
    else:
        print(out)
        print(grad)
        print('Error, check computed gradients')


if __name__ == "__main__":

    def a(x1, x2):
        return ops.ln(x1) + x1*x2 - ops.sin(x2)

    def b(x1, x2):
        return x1*x2 - ops.exp(x1-x2)*ops.sin(x1)

    def c(x1, x2):
        return ops.exp(x1)*ops.sin(x2) + ops.exp(x2)*ops.cos(x1)

    def d(x1, x2):
        return x1*(ops.exp(-(x1**2 + x2**2)))

    function_test(a, [2, 5], [11.652071455223084, [5.5, 1.7163378145367738]])

    function_test(b, [3, 2], [5.616396046458869, [
                  4.307474660278663, 3.383603953541131]])

    function_test(c, [6, -2], [-366.70681891263416, [
                  -366.79894905474396, -167.75567126633288]])

    function_test(d, [-1, 3], [-4.5399929762484854e-05, [
                  -4.5399929762484854e-05, 0.0002723995785749091]])
