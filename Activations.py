import numpy as np
from Tensor import Tensor

# Turn these into classes


def ReLU(x):
    output = Tensor(np.maximum(x.value, 0),
                    children=(x,), fun='ReLU')

    def _backward():
        x.grad += output.grad*(output.value)

    output._backward = _backward

    return output


if __name__ == "__main__":

    x = Tensor([10, -1])
    y = ReLU(x)
    z = y.sum()
    print(z)
    z.backward()
    print(x.grad)
    # print(y.grad)
