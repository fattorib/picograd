import numpy as np
from MiniNN.Tensor import Tensor
from MiniNN.Loss import MSELoss


class Linear():

    def __init__(self, in_feats, out_feats, bias=True):
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weights = Tensor.random_uniform(self.in_feats, self.out_feats)
        self.bias_flag = bias

        if self.bias_flag:
            self.bias = Tensor.random_uniform(1, self.out_feats)

    def forward(self, input):
        return input.dot(self.weights)+self.bias

    def __call__(self, input):
        return input.dot(self.weights)+self.bias


class Dropout():

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if self. p > 0:
            dropout_vals = np.random.binomial(
                [np.ones(input.shape)], 1-self.p)[0]

            val = input.value*dropout_vals*(1/(1-self.p))

            output = Tensor(val,
                            children=(input,), fun='DropoutBackward')

            def _backward():
                # Same issue as ReLU
                input.grad += output.grad*((val != 0)*(1/(1-self.p)))

            output._backward = _backward

            return output

        else:
            return input

# -------Activations


class ReLU():
    @staticmethod
    def __call__(input):

        val = np.maximum(input.value, 0)
        output = Tensor(val,
                        children=(input,), fun='ReLUBackward')

        def _backward():
            # These gotta be ones!!!!
            input.grad += output.grad*((val > 0).astype(np.float32))

        output._backward = _backward

        return output


class Sigmoid():

    @staticmethod
    def __call__(input):
        # Disable overflow warnings
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')

            # These cases are needed, overflow warning else
            val = np.where(input.value >= 0,
                           1/(1 + np.exp(-input.value)),
                           np.exp(input.value)/(1 + np.exp(input.value))
                           )
        output = Tensor(val,
                        children=(input,), fun='SigmoidBackard')

        def _backward():
            input.grad += output.grad*(val*(1-val))

        output._backward = _backward

        return output


class Tanh():
    @staticmethod
    def __call__(input):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')

            val = np.tanh(input.value)
        output = Tensor(val,
                        children=(input,), fun='TanhBackard')

        def _backward():
            input.grad += output.grad*(1-(val**2))

        output._backward = _backward

        return output

# --------Softmaxes


class Softmax():
    @ staticmethod
    def __call__(input, dim):
        exp = np.exp(input.value)
        sum = np.sum(np.exp(input.value), 1)
        sum = np.expand_dims(sum, 1)
        val = exp/sum
        output = Tensor(val,
                        children=(input,), fun='SoftmaxBackward')

        def _backward():
            R_bar = -np.sum(output.grad*exp, axis=1, keepdims=True)/(sum**2)
            input.grad += (output.grad)*(exp/sum) + R_bar*exp

        output._backward = _backward

        return output


class LogSoftmax():

    @ staticmethod
    def __call__(input, dim):

        exp = np.exp(input.value)
        sum = np.sum(np.exp(input.value), 1)
        sum = np.expand_dims(sum, 1)
        val = np.log(exp/sum)

        output = Tensor(val,
                        children=(input,), fun='LogSoftmaxBackward')

        # Complete credit to TinyGrad for this gradient...
        def _backward():
            input.grad += (output.grad)-(np.exp(val) *
                                         (np.sum(output.grad, axis=1).reshape((-1, 1))))

        output._backward = _backward

        return output


# ---------Convolutional Layers---------
# lol


if __name__ == "__main__":

    x = Tensor([[0.001, 0.2, -1, 3], [3.2, 1, 3, 1]])

    softmax = Softmax()

    y = softmax(x, dim=1)

    print(y)

    actual_out = Tensor([[1, 2, 3, 4], [3.2, 1, 3, 1]])

    mse = MSELoss()
    loss = mse(y, actual_out)
    loss.backward()
    print(x.grad)

    import torch

    x = torch.tensor([[0.001, 0.2, -1, 3], [3.2, 1, 3, 1]], requires_grad=True)
    softmax = torch.nn.Softmax(dim=1)
    y = softmax(x)
    print(y)

    actual_out = torch.tensor(
        [[1, 2, 3, 4], [3.2, 1, 3, 1]], requires_grad=True)
    mse = torch.nn.MSELoss()
    loss = mse(y, actual_out)
    loss.backward()
    print(x.grad)
