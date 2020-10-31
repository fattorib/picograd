
class Node():
    def __init__(self, value, fun, *args):
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
        self.parents = []
        self.fun = fun
        self.other = args
