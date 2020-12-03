from Tensor import *

from nn import Linear, ReLU, LogSoftmax

from Loss import NLLLoss


class Network():

    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)
        self.logsoftmax = LogSoftmax()
        self.relu = ReLU()

    def forward(self, input):
        x = self.relu(self.fc1(input))
        x = self.fc2(x)
        return self.logsoftmax(x, 1)


model = Network()

x_init = np.random.randn(2, 784).astype(np.float32)

single_batch = Tensor(x_init)

print(model.forward(single_batch))
