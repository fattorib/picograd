
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
        # Using the same keys as referenced in graph should make later querying easier
        self.parents = []
        self.fun = fun
        self.other = args
