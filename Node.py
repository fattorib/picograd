
class Node():
    def __init__(self, value, fun, graph, requires_grad, *args):
        """

        Parameters
        ----------
        value : np.array(float64)
            Value at node
        parents : Node
            Parent node(s)
        fun : str
            Primitive function at node
        *args : ??
            ???
        Returns
        -------
        None.

        """

        self.value = value
        # Using the same keys as referenced in graph should make later querying easier
        self.parents = []
        self.fun = fun
        self.other = args

        # The ol' pytorch meme
        self.requires_grad = requires_grad
        # Used for operator overloading
        self.graph = graph

    def __repr__(self):
        return str(self.value)

    # Non-Overloaded operators to be called by overloaded operators

    def scale(self, n):

        v = n*(self.value)

        # Create new node
        v_i = Node(v, 'Scaling', self.graph, False, n)

        # Add it to existing graph
        v_idx = self.graph(v_i)

        # Tell a and b that v_i node is a parent
        self.parents.append(v_idx)

        return v_i

    def recip(self):

        # Add check to ensure no division by 0
        v = 1/self.value

        # Create new node
        v_i = Node(v, 'Reciprocal', self.graph, False)

        # Add it to existing graph
        v_idx = self.graph(v_i)

        # Tell a and b that v_i node is a parent
        self.parents.append(v_idx)

        return v_i

    # Overloaded operators

    def __add__(self, b):

        if type(b) == Node or type(b) == Variable:
            v = self.value+b.value

            # Create new node
            v_i = Node(v, 'Addition', self.graph, False)

            # Add it to existing graph
            v_idx = self.graph(v_i)

            # Tell a and b that v_i node is a parent
            self.parents.append(v_idx)
            b.parents.append(v_idx)

            return v_i

        else:
            # Shifting by a scalar
            v = self.value+b

            # Create new node
            v_i = Node(v, 'Shifting', self.graph, False)

            # Add it to existing graph
            v_idx = self.graph(v_i)

            # Tell a that v_i node is a parent
            self.parents.append(v_idx)

            return v_i

    def __radd__(self, b):
        if type(b) == Node or type(b) == Variable:
            v = self.value+b.value

            # Create new node
            v_i = Node(v, 'Addition', self.graph, False)

            # Add it to existing graph
            v_idx = self.graph(v_i)

            # Tell a and b that v_i node is a parent
            self.parents.append(v_idx)
            b.parents.append(v_idx)

            return v_i

        else:
            # Shifting by a scalar
            v = self.value+b

            # Create new node
            v_i = Node(v, 'Shifting', self.graph, False)

            # Add it to existing graph
            v_idx = self.graph(v_i)

            # Tell a that v_i node is a parent
            self.parents.append(v_idx)

            return v_i

    def __pow__(self, n):
        v = (self.value)**n

        # Create new node
        v_i = Node(v, 'Power', self.graph, False, n)

        # Add it to existing graph
        v_idx = self.graph(v_i)

        # Tell a and b that v_i node is a parent
        self.parents.append(v_idx)

        return v_i

    def __neg__(self):
        v = -1*self.value

        # Create new node
        v_i = Node(v, 'Negative', self.graph, False)

        # Add it to existing graph
        v_idx = self.graph(v_i)

        # Tell a and b that v_i node is a parent
        self.parents.append(v_idx)

        return v_i

    def __mul__(self, b):

        if type(b) == Node or type(b) == Variable:
            v = self.value*b.value

            # Create new node
            v_i = Node(v, 'Multiplication', self.graph, False)

            # Add it to existing graph
            v_idx = self.graph(v_i)

            # Tell a and b that v_i node is a parent
            self.parents.append(v_idx)
            b.parents.append(v_idx)

            return v_i

        else:
            return self.scale(b)

    def __rmul__(self, b):

        if type(b) == Node or type(b) == Variable:
            v = self.value*b.value

            # Create new node
            v_i = Node(v, 'Multiplication', self.graph, False)

            # Add it to existing graph
            v_idx = self.graph(v_i)

            # Tell a and b that v_i node is a parent
            self.parents.append(v_idx)
            b.parents.append(v_idx)

            return v_i

        else:
            return self.scale(b)

    # Implement rsub too

    def __sub__(self, b):
        return self + (-b)

    def __rsub__(self, b):
        return self + (-b)

    def __truediv__(self, b):

        return self*(b.recip())

    def __rtruediv__(self, b):

        return self.recip()*(b)


class Variable(Node):
    def __init__(self,
                 value, graph, fun='Leaf', requires_grad=True, * args):
        super().__init__(value, graph, fun, requires_grad, *args)
        # Really the only change
        """
        Difference between this and node is that the gradient is automatically tracked.
        """

        self.fun = fun
        self.value = value
        # Using the same keys as referenced in graph should make later querying easier
        self.parents = []
        self.other = args

        # Used for operator overloading
        self.graph = graph
        self.graph(self)
