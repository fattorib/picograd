# loss functions go here
from Tensor import Tensor
import numpy as np


class MSELoss():
    @staticmethod
    def __call__(y, y_hat):
        return ((y-y_hat)**2).mean()


# class CrossEntropyLoss():


if __name__ == "__main__":

    x = Tensor([[3, -0.5, 2, 7], [3, -0.5, 2, 7]])
    y = Tensor([[2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8]])

    print(x.shape)

    mse = MSELoss()

    z = mse(x, y)
    z.backward()
    print(z)
    print(x.grad)
    print(y.grad)
