
class Node():
    def __init__(self, value, parent, fun, *args):
        """

        Parameters
        ----------
        value : np.array(float64)
            Value at node
        parent : Node
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
        self.parents = parents
        self.fun = fun
        self.other = args
