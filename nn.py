import numpy as np
from Tensor import Tensor


class Linear():

    def __init__(self, in_feats, out_feats, bias=True):
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weights = Tensor.random_uniform(self.in_feats, self.out_feats)
        self.bias_flag = bias

        if self.bias_flag:
            self.bias = Tensor.random_uniform(1, self.out_feats)

    def forward(self, input):
        return input.dot(self.weights)+self.bias

    def __call__(self, input):
        return input.dot(self.weights)+self.bias


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
    def __call__(input, axis):
        exp = input.exp()
        sum = (input.exp()).sum(axis)
        print(exp.shape)
        sum.expand_dim(axis)
        return exp/sum


if __name__ == "__main__":

    x = Tensor.ones((10, 2))

    softmax = Softmax()

    print(softmax(x, axis=1))
