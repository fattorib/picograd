import numpy as np

from Node import *


class Computational_Graph():
    """
    Class holding the actual computational graph. This is updated with every
    pass adding a new node
    """

    def __init__(self):
        self.comp_graph = []

    def __call__(self, node):
        self.comp_graph.append(node)

    def graph_visualize_list(self):
        # Helpful function for debugging
        for i in range(0, len(self.comp_graph)):
            print('v_' + str(i-1),
                  self.comp_graph[i].fun, self.comp_graph[i].value)


class array():
    """
    Class for holding our gradient-tracking arrays. Wrapping a numpy array. 
    Implementation might change significantly
    """

    def __init__(self, data):
        self.values = np.array([i for i in data])
        self.vec_len = len(data)
        self.grad = np.zeros(self.vec_len)
        self.graph = []
        self.dtype = self.values.dtype
