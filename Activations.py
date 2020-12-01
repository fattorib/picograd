import numpy as np
from Tensor import Tensor


class ReLU():
    # Missing init method here

    @staticmethod
    def __call__(input):
        output = Tensor(np.maximum(input.value, 0),
                        children=(input,), fun='ReLU')

        def _backward():
            input.grad += output.grad*(output.value)

        output._backward = _backward

        return output


class Sigmoid():

    @staticmethod
    def __call__(input):

        val = 1/(1+np.exp(-(input.value)))

        output = Tensor(val,
                        children=(input,), fun='Sigmoid')

        def _backward():
            input.grad += output.grad*(val*(1-val))

        output._backward = _backward

        return output


class Softmax():

    @staticmethod
    def __call__(input):
        val = np.exp(input.value)/np.sum(np.exp(input.value))

        output = Tensor(val,
                        children=(input,), fun='Softmax')

        def _backward():
            input.grad += output.grad*(val*(1-val))

        output._backward = _backward

        return output


def sm(x):
    return x.exp()/(x.exp().sum(1))


if __name__ == "__main__":

    # Tinygrad example. Works!
    x = Tensor.eye(3)
    y = Tensor([[2.0, 0, -2.0]])
    z = y.dot(x).sum(1)
    z.backward()

    print(x.grad)  # dz/dx
    print(y.grad)  # dz/dy
