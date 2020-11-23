import numpy as np
from Node import *


class Tensor():
    def __init__(self, values, graph, requires_grad, *args):

        # Add some code to convert values to an np.array for better indexing,etc
        # Assume 1-D list
        if type(values) == list:
            if requires_grad == True:
                self.arr = np.array([Variable(i, graph) for i in values])
                self.value = np.array(values)
            # If no grad, assume this is intermediate tensor from some vector computation.
            # In this case, we are passing in array of nodes. We can get
            # the values at the same time
            else:
                self.arr = np.array([i for i in values])
                self.value = np.array([i for i in values])

        else:
            # Assuming multidimensional np array
            if requires_grad == True:
                self.arr = np.array([Variable(i, graph)
                                     for i in values.flatten()]).reshape(values.shape)
                self.value = np.array(values)

            # Assume this is intermediate tensor from some vector computation.
            # In this case, assume we are passing in array of nodes. We can get
            # the values at the same time
            else:
                self.arr = np.array([i for i in values])
                self.value = np.array(
                    [i.value for i in values.flatten()]).reshape(values.shape)

        self.graph = graph

        self.len = len(values)
        self.shape = self.arr.shape

    def grad(self):
        # This is needed as the gradient is not updated if it is created at first init of tensor
        return [self.arr[i].grad for i in range(0, len(self.arr))]

    def __len__(self):
        return self.len

    def __repr__(self):
        return str(self.value)

    def scale(self, n):
        node_arr = []
        for i in self.arr:
            i = n*i
            node_arr.append(i)

        return Tensor(node_arr, self.graph, False)

    # Overloading
    def __add__(self, other):
        if type(other) == Tensor:
            try:
                other.shape[1]
                assert self.shape == other.shape, 'Broadcasting not supported :('
                nodes_arr = []
                for i, j in zip(self.arr.flatten(), other.arr.flatten()):
                    nodes_arr.append(i+j)
                nodes_arr = np.array(nodes_arr).reshape(self.shape)
                return Tensor(nodes_arr, self.graph, False)

            except IndexError:
                nodes_arr = []
                for i, j in zip(self.arr, other.arr):
                    nodes_arr.append(i+j)

                return Tensor(nodes_arr, self.graph, False)

        elif type(other) == list:
            nodes_arr = []
            for i, j in zip(self.arr, other):
                nodes_arr.append(i+j)
            nodes_arr = np.array(nodes_arr).reshape(self.shape)
            return Tensor(nodes_arr, self.graph, False)

        else:
            # Broadcasting when adding scalar
            nodes_arr = []
            for i in self.arr:
                nodes_arr.append(i+other)

            return Tensor(nodes_arr, self.graph, False)

    def __radd__(self, other):
        if type(other) == Tensor:
            nodes_arr = []
            for i, j in zip(self.arr, other.arr):
                nodes_arr.append(i+j)

            return Tensor(nodes_arr, self.graph, False)

        elif type(other) == list:
            nodes_arr = []
            for i, j in zip(self.arr, other):
                nodes_arr.append(i+j)
            nodes_arr = np.array(nodes_arr).reshape(self.shape)
            return Tensor(nodes_arr, self.graph, False)

        else:
            # Broadcasting
            nodes_arr = []
            for i in self.arr:
                nodes_arr.append(i+other)

            return Tensor(nodes_arr, self.graph, False)

    def __mul__(self, other):
        if type(other) == Tensor:
            assert len(self) == len(
                other), "Check your input tensors, lengths do not match"
            nodes_arr = []
            for i, j in zip(self.arr, other.arr):
                nodes_arr.append(i*j)

            return Tensor(nodes_arr, self.graph, False)

        else:
            # Assuming scalar mult
            return self.scale(other)

    def __rmul__(self, other):
        if type(other) == Tensor:
            # Perform Hadamard Product
            assert len(self) == len(
                other), "Check your input tensors, lengths do not match"
            nodes_arr = []
            for i, j in zip(self.arr, other.arr):
                nodes_arr.append(i*j)

            return Tensor(nodes_arr, self.graph, False)

        else:
            # Assuming scalar mult
            return self.scale(other)

    def __pow__(self, n):
        nodes_arr = []
        for i in self.arr:
            nodes_arr.append(i**2)
        return Tensor(nodes_arr, self.graph, False)

    def __sub__(self, other):
        if type(other) == Tensor:
            try:
                # Need to either: include broadcast commands or disallow it
                other.shape[1]
                assert self.shape == other.shape, 'Broadcasting currently not supported :('
                nodes_arr = []
                for i, j in zip(self.arr.flatten(), other.arr.flatten()):
                    nodes_arr.append(i-j)
                nodes_arr = np.array(nodes_arr).reshape(self.shape)
                return Tensor(nodes_arr, self.graph, False)

            except IndexError:
                nodes_arr = []
                for i, j in zip(self.arr, other.arr):
                    nodes_arr.append(i-j)

                return Tensor(nodes_arr, self.graph, False)

        elif type(other) == list:
            nodes_arr = []
            for i, j in zip(self.arr, other):
                nodes_arr.append(i-j)
            nodes_arr = np.array(nodes_arr).reshape(self.shape)
            return Tensor(nodes_arr, self.graph, False)

        else:
            # Broadcasting when adding scalar
            nodes_arr = []
            for i in self.arr:
                nodes_arr.append(i-other)

            return Tensor(nodes_arr, self.graph, False)

    def __rsub__(self, other):
        if type(other) == Tensor:
            try:
                other.shape[1]
                assert self.shape == other.shape, 'Broadcasting currently not supported :('
                nodes_arr = []
                for i, j in zip(self.arr.flatten(), other.arr.flatten()):
                    nodes_arr.append(i-j)
                nodes_arr = np.array(nodes_arr).reshape(self.shape)
                return Tensor(nodes_arr, self.graph, False)

            except IndexError:
                nodes_arr = []
                for i, j in zip(self.arr, other.arr):
                    nodes_arr.append(i-j)

                return Tensor(nodes_arr, self.graph, False)

        elif type(other) == list:
            nodes_arr = []
            for i, j in zip(self.arr, other):
                nodes_arr.append(i-j)
            nodes_arr = np.array(nodes_arr).reshape(self.shape)
            return Tensor(nodes_arr, self.graph, False)

        else:
            # Broadcasting when adding scalar
            nodes_arr = []
            for i in self.arr:
                nodes_arr.append(i-other)

            return Tensor(nodes_arr, self.graph, False)
