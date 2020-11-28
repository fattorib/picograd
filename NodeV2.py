class Node():
    def __init__(self, value, fun='', children=(), *args):
        """

        Parameters
        ----------
        value : np.array(float64)
            Value at node
        children : Node
            Children node(s). Note that some references will call this node the 'Parent'
        fun : str
            Primitive function at node
        Returns
        -------
        None.

        """
        self.value = value
        self.children = set(children)
        self.fun = fun
        self.other = args
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return str(self.value)

    def __add__(self, other):
        # For now, we assume that the other value is also a node too
        # Can add a catch for this later on
        output = Node(self.value + other.value,
                      children=(self, other), fun='add')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward

        return output

    def __mul__(self, other):
        # For now, we assume that the other value is also a node too
        # Can add a catch for this later on
        output = Node(self.value*other.value,
                      children=(self, other), fun='mul')

        def _backward():
            self.grad += output.grad*other.value
            other.grad += output.grad*self.value

        output._backward = _backward

        return output

    def backward(self):
        # Compute the backward pass starting at this node

        # Always assume the the base gradient is 1
        self.grad = 1

        # Build the computational graph

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
