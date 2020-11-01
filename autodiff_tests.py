import numpy as np
from Node import *
import Computational_Graph as G
import Grad_ops as ops


"""
Single Variable Function Tests
"""


def function_test_1var(func, value, expected_output):
    # Init computational graph
    graph = G.Computational_Graph()
    values_array = []
    for i in value:
        k = Node(i, 'Leaf', graph)
        values_array.append(k)
        graph(k)

    out = func(values_array[0]).value
    grad = graph.backward()
    if out == expected_output[0] and grad == expected_output[1]:
        print('Test passed')
    else:
        print(out)
        print(grad)
        print('Error, check computed gradients')


"""
Two Variable Function Tests
"""


def function_test_2var(func, values, expected_outputs):
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
        print(out[0])
        print(grad)
        print('Error, check computed gradients')


if __name__ == "__main__":

    def a(x):
        return (ops.exp(x)+1).recip()

    def b(x):
        return (ops.exp(x*2) - 1)/(ops.exp(x*2) + 1)

    def c(x):
        return ops.ln(x) + ops.exp(x**2)

    def d(x):
        return ops.sin(ops.cos(ops.ln(x) + x**2))

    print('Single Variable Tests:')
    function_test_1var(a, [0], [0.5, [-0.25]])
    function_test_1var(b, [1], [0.7615941559557649, [0.419974341614026]])
    function_test_1var(c, [np.pi], [19334.833804250986, [121477.46943551568]])
    function_test_1var(
        d, [1], [0.5143952585235492, [-2.1648184471903296]])

    def a(x1, x2):
        return ops.ln(x1) + x1*x2 - ops.sin(x2)

    def b(x1, x2):
        return x1*x2 - ops.exp(x1-x2)*ops.sin(x1)

    def c(x1, x2):
        return ops.exp(x1)*ops.sin(x2) + ops.exp(x2)*ops.cos(x1)

    def d(x1, x2):
        return x1*(ops.exp(-(x1**2 + x2**2)))

    print()
    print('Two Variable Tests:')
    function_test_2var(
        a, [2, 5], [11.652071455223084, [5.5, 1.7163378145367738]])

    function_test_2var(b, [3, 2], [5.616396046458869, [
        4.307474660278663, 3.383603953541131]])

    function_test_2var(c, [6, -2], [-366.70681891263416, [
        -366.79894905474396, -167.75567126633288]])

    function_test_2var(d, [-1, 3], [-4.5399929762484854e-05, [
        -4.5399929762484854e-05, 0.0002723995785749091]])
