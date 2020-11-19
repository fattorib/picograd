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

        if node.requires_grad == True:
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

    def zero_gradients(self):
        """
        Call before every gradient pass so gradients don't accumulate
        """
        # self.comp_graph_grad.clear()
        to_pop = []
        for node_idx in self.comp_graph:
            node = self.comp_graph.get(node_idx)
            if node.fun != 'Leaf':
                to_pop.append(node_idx)
            else:
                # Required or else, gradients will accumulate
                node.parents = []

        [self.comp_graph.pop(node_key, None) for node_key in to_pop]
        [self.comp_graph_grad.pop(node_key, None) for node_key in to_pop]

    def backward(self):
        """
        Perform backpropogation to compute leaf gradients
        """
        output_dim = 0
        for k in reversed(list(self.comp_graph)):
            node = self.comp_graph.get(k)

            assert output_dim < 2, "Gradient computations only supported for vector to scalar functions!"

            if node.parents == []:
                output_dim += 1
                pass
            else:
                # List
                parents = node.parents

                node_grad = 0

                # Numerical Value
                value = node.value

                # Looping through each parent node
                for parent in parents:
                    # Get the actual node itself
                    parent_node = self.comp_graph.get(parent)
                    parent_grad = self.comp_graph_grad.get(parent)

                    # Getting the function at the parent node
                    fun = parent_node.fun
                    fun_lambda = dict_functions[fun]

                    # Add a divide by 0 catch here
                    if value == 0:
                        parent_single_val = parent_grad * \
                            fun_lambda(value, parent_node.value,
                                       parent_node.other)
                    else:
                        parent_single_val = parent_grad * \
                            fun_lambda(value, parent_node.value /
                                       value, parent_node.other)

                    node_grad += parent_single_val

                # Update node gradient value
                self.comp_graph_grad[k] = node_grad

        # Go through leaves and pull their gradients
        gradient = []
        for i in self.comp_leaves:
            gradient.append(self.comp_graph_grad.get(i))

        return gradient
