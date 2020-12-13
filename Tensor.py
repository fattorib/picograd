import numpy as np
import cupy as cp


class Tensor():
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

        if type(value) == list:
            self.value = np.array(value, dtype=np.float32)
            self.shape = self.value.shape

        if type(value) == np.ndarray:
            self.value = value
            self.shape = self.value.shape

        # This is buggy
        elif type(value) != list and type(value) != np.ndarray:
            self.value = np.array(value, dtype=np.float32)
            self.shape = (1,)

        self.children = set(children)
        self.fun = fun

        self._backward = lambda: None

        self.grad = np.zeros_like(self.value, dtype=np.float32)

        self.gpu_flag = False

    def zero_grad(self):
        self.grad = np.zeros_like(self.value, dtype=np.float32)

    def expand_dim(self, axis):
        self.value = np.expand_dims(self.value, axis)
        self.shape = self.value.shape

    def gpu(self):
        # Pass value and grad to gpu
        self.value = cp.asarray(self.value)
        self.grad = cp.asarray(self.grad)
        self.gpu_flag = True

    def __repr__(self):
        return '(' + str(self.value) + ', grad_fn =<' + self.fun + '>)'

    def __add__(self, other):
        if type(other) != Tensor:
            other = Tensor(other)

        output = Tensor(self.value + other.value,
                        children=(self, other), fun='AddBackward')

        def _backward():
            # Broadcasting is unhappy with +=
            self.grad = self.grad + output.grad
            other.grad = other.grad + output.grad

        output._backward = _backward

        return output

    def __sub__(self, other):
        if type(other) != Tensor:
            other = Tensor(other)

        output = Tensor(self.value - other.value,
                        children=(self, other), fun='SubBackward')

        def _backward():
            self.grad += output.grad
            other.grad += -output.grad

        output._backward = _backward

        return output

    def __mul__(self, other):
        if type(other) != Tensor:
            other = Tensor(other)

        output = Tensor(self.value*other.value,
                        children=(self, other), fun='MulBackward')

        def _backward():
            # These are problematic
            self.grad += (output.grad)*other.value
            other.grad += (output.grad)*self.value

        output._backward = _backward

        return output

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __neg__(self):
        return self*-1.0

    def __pow__(self, other):
        output = Tensor(self.value**other, children=(self,), fun='PowBackward')

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
        assert (
            self.shape == () or (self.shape[0] == 1)), "Backward pass only supported for vector to scalar functions"

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

    def sum(self, *axis):
        output = Tensor(np.sum(self.value, *axis),
                        children=(self,), fun='SumBackward')

        # Backward pass might be incorrect now. Yeah this isn't correct
        def _backward():
            # self.grad += np.ones_like(self.value)*output.grad
            self.grad += output.grad

        output._backward = _backward

        return output

    def mean(self, *axis):
        output = Tensor(np.mean(self.value, *axis),
                        children=(self,), fun='MeanBackward')

        # Backward pass might be incorrect now?
        def _backward():
            try:
                self.grad += (1/(self.shape[0]*self.shape[1]))*output.grad
            except IndexError:
                self.grad += (1/(self.shape[0]))*output.grad

        output._backward = _backward

        return output

    def dot(self, weight):

        # Case when self is input,
        output = Tensor(self.value.dot(weight.value),
                        children=(self, weight), fun='DotBackward')

        def _backward():
            self.grad += output.grad.dot(np.transpose(weight.value))
            weight.grad += np.transpose(self.value).dot(output.grad)

        output._backward = _backward

        return output

    def matmul(self, weight):
        output = Tensor(np.matmul(self.value, (weight.value)),
                        children=(self, weight), fun='MatmulBackard')

        def _backward():
            self.grad += np.matmul(output.grad, np.transpose(weight.value))
            weight.grad += np.matmul(np.transpose(self.value), (output.grad))

        output._backward = _backward

        return output

    def norm(self):
        # Euclidean norm
        return ((self**2).sum())**(1/2)

    def exp(self):
        output = Tensor(np.exp(self.value), children=(self,), fun='ExpBackard')

        def _backward():
            self.grad += output.grad*np.exp(self.value)

        output._backward = _backward

        return output

    def log(self):
        output = Tensor(np.log(self.value), children=(self,), fun='LogBackard')

        def _backward():
            self.grad += output.grad*(self.value)**(-1)

        output._backward = _backward

        return output

    @ staticmethod
    def zeros(shape):
        return Tensor(np.zeros(shape, dtype=np.float32))

    @ staticmethod
    def ones(shape):
        return Tensor(np.ones(shape, dtype=np.float32))

    @ staticmethod
    def random(*shape):
        return Tensor(np.random.rand(*shape).astype(np.float32))

    @ staticmethod
    def eye(shape):
        return Tensor(np.eye(shape, dtype=np.float32))

    @ staticmethod
    def random_uniform(*shape):

        random_vals = np.random.uniform(-1, 1,
                                        size=shape)/np.sqrt(np.prod(shape))
        # random_vals = np.random.uniform(-np.sqrt(np.prod(shape)), np.sqrt(np.prod(shape)),
        #                                 size=shape)

        return Tensor(random_vals.astype(np.float32))
