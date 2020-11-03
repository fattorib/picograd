# Implement common functionals
from Node import *
from Computational_Graph import *
import Grad_ops as ops
import numpy as np


def tanh(x):
    return (ops.exp(2*x) - 1)/(ops.exp(2*x) + 1)


def sigmoid(x):
    return (1 + ops.exp(-x)).recip()


if __name__ == "__main__":
    graph = Computational_Graph()

    x = Variable(2, graph)
    z = 1/(1 + np.exp(-2))

    print(sigmoid(x).value == z)

    print(tanh(x).value == np.tanh(2))
