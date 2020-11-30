import numpy as np
from Tensor import Tensor


def exp(x):
    output = Tensor(np.exp(x.value), children=(x,), fun='exp')

    def _backward():
        x.grad += np.exp(x.value)*output.grad

    output._backward = _backward

    return output


def sin(x):
    output = Tensor(np.sin(x.value), children=(x,), fun='sin')

    def _backward():
        x.grad += np.cos(x.value)*output.grad

    output._backward = _backward

    return output


def cos(x):
    output = Tensor(np.cos(x.value), children=(x,), fun='cos')

    def _backward():
        x.grad += -np.sin(x.value)*output.grad

    output._backward = _backward

    return output


def log(x):
    output = Tensor(np.log(x.value), children=(x,), fun='log')

    def _backward():
        x.grad += (x.value**-1)*output.grad

    output._backward = _backward

    return output

# Vector -> Vector functions

# Implementing for Matrix-Vector Product
# x is matrix, y is vector

# Makes sense to package this in a Linear class


def linear_mm(weights, input):
    output = Tensor(np.matmul(input.value, np.transpose(
        weights.value)), children=(weights, input), fun='linear')

    def _backward():
        # Following 'Linear' code from: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
        weights.grad += np.matmul(np.transpose(output.grad), input.value)
        input.grad += np.matmul(output.grad, weights.value)

    output._backward = _backward

    return output

# Vector -> Scalar functions


def dot(x, y):
    output = Tensor(np.dot(x.value, y.value), children=(x, y), fun='dot')

    def _backward():
        x.grad += y.value*output.grad
        y.grad += x.value*output.grad

    output._backward = _backward

    return output


def sum(x):
    output = Tensor(np.sum(x.value), children=(x,), fun='sum')

    def _backward():
        x.grad += output.grad

    output._backward = _backward

    return output


if __name__ == "__main__":
    out_feats = 10
    in_feats = 20
    weight_matrix = Tensor.random(out_feats, in_feats)
    input_vector = Tensor.ones((1, in_feats))

    c = mm(weight_matrix, input_vector)
    d = sum(c)
    d.backward()
    # print(weight_matrix.grad)
    print(input_vector.grad)
