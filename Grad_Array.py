import numpy as np

from Node import *


class Computational_Graph():
    """
    Class holding the actual computational graph. This is updated with every
    pass adding a new node
    """

    def __init__(self):
        self.comp_graph = {}
        self.comp_graph_grad = {}

    def __call__(self, node):
        """
        Used to add new node to graph
        """
        v_idx = 'v_'+str(len(self.comp_graph)-1)
        self.comp_graph[v_idx] = node
        self.comp_graph_grad[v_idx] = 1

        return v_idx

    def graph_visualize_list(self):
        # Helpful function for debugging
        for i in self.comp_graph:
            node = self.comp_graph.get(i)
            print(i, node.fun,
                  node.value,
                  node.parents
                  )

    def backward(self):
        """
        The actual hard part of this all. Go backward through the graph
        and properaly compute the gradient for all leaf nodes
        """
        for k in reversed(list(self.comp_graph)):
            node = self.comp_graph.get(k)
            if node.parents == []:
                # Root node, don't need to adjust gradient
                print('Root node')
            else:
                parents = node.parents
                for parent in parents:
                    parent_node = self.comp_graph.get(parent)
                    print(parent_node.fun, parent)


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
