import numpy as np


class Node():
    def __init__(self, value, fun='', children=()):
        """

        Parameters
        ----------
        value : np.array(float64)
            Value at node
        children : Node
            Children node(s). Note that some references will call this node the 'Parent'
        fun : str
            Primitive function at node
        grad: float
            gradient value of the node. defaults to 0

        _backward: lambda
            gradient computation at this node

        Returns
        -------
        None.

        """

        # Wrap everything in numpy arrays
        # self.value = np.array(value)
        if type(value) == list:
            self.value = np.array(value)
            self.shape = self.value.shape
        else:
            self.value = np.array(value)
            self.shape = (1,)

        self.children = set(children)
        self.fun = fun

        self._backward = lambda: None

        self.grad = np.zeros_like(self.value)

    def __repr__(self):
        return str(self.value)

    def __add__(self, other):
        if type(other) != Node:
            other = Node(other)

        output = Node(self.value + other.value,
                      children=(self, other), fun='add')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward

        return output

    def __mul__(self, other):
        if type(other) != Node:
            other = Node(other)

        output = Node(self.value*other.value,
                      children=(self, other), fun='mul')

        def _backward():
            self.grad += output.grad*other.value
            other.grad += output.grad*self.value

        output._backward = _backward

        return output

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return -1*self

    def __pow__(self, other):
        output = Node(self.value**other, children=(self,), fun='pow')

        def _backward():
            self.grad += (other)*(self.value**(other-1))*output.grad

        output._backward = _backward

        return output

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self*other

    def backward(self):
        # Compute the backward pass starting at this node

        # Always assume the the base gradient is 1
        assert self.shape[0] == 1, "Backward pass only supported for vector to scalar functions"

        self.grad = np.array(1)

        # Build the computational graph. Using Karpathy Micrograd fn
        visited_nodes = set()
        topo_sorted_graph = []

        def build_graph(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child_nodes in node.children:
                    build_graph(child_nodes)
                topo_sorted_graph.append(node)

        build_graph(self)

        for node in reversed(topo_sorted_graph):
            node._backward()


if __name__ == "__main__":
    print('Hey')
