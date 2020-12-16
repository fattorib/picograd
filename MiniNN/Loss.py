# loss functions go here
from Tensor import Tensor
import numpy as np
# from nn import LogSoftmax


class MSELoss():
    @staticmethod
    def __call__(y, y_hat):
        return ((y-y_hat)**2).mean()


# class CrossEntropyLoss():
#     @staticmethod
#     def __call__(x, C):
#         log = LogSoftmax()
#         return None


class NLLLoss():
    @staticmethod
    def __call__(x, classes):

        # Get shape of classes
        scalar = classes.shape[1]
        classes.value = -scalar*classes.value

        return (x*((classes))).mean()


# class CrossEntropyLoss():
if __name__ == "__main__":

    x = Tensor([[10, -0.19], [-0.38, 1.99], [0.10, 0.122]])
    y = Tensor([[0, 1], [1, 0], [1, 0]])

    print(x.shape)

    mse = MSELoss()

    z = mse(x, y)
    print(z)
    z.backward()
    print(x.grad)

    import torch

    x = torch.tensor([[10, -0.19], [-0.38, 1.99],
                      [0.10, 0.122]], requires_grad=True)

    y = torch.tensor([1.0, 0.0, 0.0], requires_grad=True).type(torch.long)
    print(y.shape)

    mse = torch.nn.MSELoss()
    z = mse(x, y)
    print(z)
    z.backward()
    print(x.grad)
