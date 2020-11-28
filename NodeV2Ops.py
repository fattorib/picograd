import numpy as np
from NodeV2 import Node


def exp(x):
    output = Node(np.exp(x.value), children=(x,), fun='exp')

    def _backward():
        x.grad += np.exp(x.value)*output.grad

    output._backward = _backward

    return output


def sin(x):
    output = Node(np.sin(x.value), children=(x,), fun='sin')

    def _backward():
        x.grad += np.cos(x.value)*output.grad

    output._backward = _backward

    return output


def cos(x):
    output = Node(np.cos(x.value), children=(x,), fun='cos')

    def _backward():
        x.grad += -np.sin(x.value)*output.grad

    output._backward = _backward

    return output


def log(x):
    output = Node(np.log(x.value), children=(x,), fun='log')

    def _backward():
        x.grad += (x.value**-1)*output.grad

    output._backward = _backward

    return output


# Vector -> Scalar functions
def dot(x, y):
    output = Node(np.dot(x.value, y.value), children=(x, y), fun='dot')

    def _backward():
        x.grad += y.value*output.grad
        y.grad += x.value*output.grad

    output._backward = _backward

    return output


def sum(x):
    output = Node(np.sum(x.value), children=(x,), fun='sum')

    def _backward():
        x.grad += output.grad

    output._backward = _backward

    return output


if __name__ == "__main__":

    a = Node([1, 2])

    b = Node([3, 2])

    c = dot(a, b)

    c.backward()
