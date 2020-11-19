from Tensor import *
from Node import *
import Computational_Graph as G
import Tensor_ops as T
from Optimizers import SGD


def f(x):
    return T.dot(x, x)


if __name__ == "__main__":
    graph = G.Computational_Graph()
    x = Tensor([1, 2, 3], graph, requires_grad=True)
    lr = 0.1
    epochs = 100
    optimizer = SGD(graph, lr)

    for i in range(0, epochs):
        optimizer.zero_grad()

        value = f(x)

        print(value, x)
        grad = graph.backward()
        # # Update x
        optimizer.step(grad, x)

    print(value, x)
