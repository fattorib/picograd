import numpy as np

from Node import *


class Computational_Graph():
    """
    Class holding the actual computational graph. This is updated with every
    pass adding a new node
    """

    def __init__(self):
        self.comp_graph = {}

    def __call__(self, node):
        v_idx = 'v_'+str(len(self.comp_graph)-1)
        self.comp_graph[v_idx] = node
        return v_idx

    def graph_visualize_list(self):
        # Helpful function for debugging
        for i in self.comp_graph:
            print(i, self.comp_graph.get(i).value,
                  self.comp_graph.get(i).parents)

    def backward(self):
        """
        The actual hard part of this all. Go backward through the graph
        and properaly compute the gradient for all leaf nodes

        """


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
