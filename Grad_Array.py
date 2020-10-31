import numpy as np

from Node import *

from VJP_dict import *


class Computational_Graph():
    """
    Class holding the actual computational graph. This is updated with every
    pass adding a new node
    """

    def __init__(self):
        self.comp_graph = {}
        self.comp_graph_grad = {}

        # Input variables to our function
        self.comp_leaves = []

    def __call__(self, node):
        """
        Used to add new node to graph
        """
        v_idx = 'v_'+str(len(self.comp_graph)-1)
        self.comp_graph[v_idx] = node
        self.comp_graph_grad[v_idx] = 1
        if node.fun == 'Leaf':
            self.comp_leaves.append(v_idx)
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
        Perform backpropogation to compute leaf gradients
        """
        for k in reversed(list(self.comp_graph)):
            node = self.comp_graph.get(k)
            if node.parents == []:
                # Root node, don't need to adjust gradient
                pass
            else:
                # List
                parents = node.parents

                # Numerical Value
                value = node.value

                node_grad = 0

                # Looping through each parent node
                for parent in parents:
                    # Get the actual node itself
                    parent_node = self.comp_graph.get(parent)
                    parent_grad = self.comp_graph_grad.get(parent)

                    # Getting the function at the parent node
                    fun = parent_node.fun
                    fun_lambda = dict_functions[fun]

                    parent_single_val = parent_grad * \
                        fun_lambda(value, parent_node.value/value)

                    node_grad += parent_single_val

                # Update node gradient value
                self.comp_graph_grad[k] = node_grad

        # Go through leaves and pull their gradients
        gradient = []
        for i in self.comp_leaves:
            gradient.append(self.comp_graph_grad.get(i))
        return gradient


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
