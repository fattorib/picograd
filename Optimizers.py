import numpy as np
import Computational_Graph as G


class SGD():

    def __init__(self, graph, lr, *args):
        self.graph = graph
        self.lr = lr

    def zero_grad(self):
        self.graph.zero_gradients()

    def step(self, grad, x):
        for i in range(0, x.shape[0]):
            x.arr[i].value = x.arr[i].value-self.lr*grad[i]
        x.value = x.value-self.lr*np.array(grad)
