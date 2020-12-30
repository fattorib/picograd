# loss functions go here
from MiniNN.Tensor import Tensor
import numpy as np


class MSELoss():
    @staticmethod
    def __call__(y, y_hat):
        return ((y-y_hat)**2).mean()


class NLLLoss():
    @staticmethod
    def __call__(x, classes):

        # Get shape of classes
        scalar = classes.shape[1]
        classes.value = -scalar*classes.value

        return (x*((classes))).mean()
