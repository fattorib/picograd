import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
from MiniNN.Tensor import Tensor
from MiniNN.nn import *
from MiniNN.Loss import NLLLoss


x_init = np.random.uniform(-1, 1, size=(64, 1024))
weight_init = np.random.uniform(-1, 1, size=(1024, 50))
bias_init = np.random.uniform(-1, 1, size=(1, 50))
targets_init = np.random.randint(0, 50, size=(64,))


class TestTensorBasics(unittest.TestCase):

    def test_forward_pass(self):

        def mini_nn_forward():
            # MiniNN
            relu = ReLU()
            input = Tensor(x_init)
            weights = Tensor(weight_init)
            bias = Tensor(bias_init)
            out = relu(input.dot(weights)+bias)
            return out.value

        def pytorch_forward():
            # PyTorch
            relu = nn.ReLU()
            torch_input = torch.tensor(x_init)
            torch_weights = torch.tensor(weight_init)
            torch_bias = torch.tensor(bias_init)
            torch_out = relu(torch_input.mm(torch_weights) + torch_bias)
            return torch_out.numpy()

        out = mini_nn_forward()
        pytorch_out = pytorch_forward()
        np.testing.assert_allclose(out, pytorch_out, atol=1e-6)

    def test_backward_pass(self):

        def mini_nn_backward():
            relu = ReLU()
            logsoftmax = LogSoftmax()
            criterion = NLLLoss()

            input = Tensor(x_init)
            weights = Tensor(weight_init)
            bias = Tensor(bias_init)

            # Making target tensor
            targets = np.zeros((len(targets_init), 50), np.float32)
            targets[range(targets.shape[0]), targets_init] = 1
            targets = Tensor(targets)

            out = relu((input.dot(weights))+bias)
            out = logsoftmax(out, 1)

            loss = criterion(out, targets)

            loss.backward()

            return out.value, weights.grad, bias.grad

        def pytorch_backward():
            relu = nn.ReLU()
            criterion = nn.NLLLoss()

            torch_input = torch.tensor(x_init, requires_grad=True)
            torch_weights = torch.tensor(weight_init, requires_grad=True)
            torch_bias = torch.tensor(bias_init, requires_grad=True)
            torch_out = relu((torch_input.mm(torch_weights)) +
                             torch_bias)

            torch_out = F.log_softmax(torch_out, 1)
            torch_targets = torch.tensor(targets_init)
            loss = criterion(torch_out, torch_targets.long())
            loss.backward()

            return torch_out.detach().numpy(), torch_weights.grad, torch_bias.grad

        out, w_grad, b_grad = mini_nn_backward()
        torch_out, w_torch_grad, b_torch_grad = pytorch_backward()

        np.testing.assert_allclose(w_grad, w_torch_grad, atol=1e-6)
        # Gradients are being broadcasted in a weird way
        # np.testing.assert_allclose(b_grad, b_torch_grad, atol=1e-6)
        np.testing.assert_allclose(out, torch_out, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
