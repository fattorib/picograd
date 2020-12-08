import matplotlib.pyplot as plt
from Tensor import *

from nn import Linear, ReLU, LogSoftmax, Sigmoid, Tanh

from Loss import NLLLoss

from optim import SGD, Adam

from GetMNIST import fetch_mnist


class Network():

    def __init__(self):
        self.fc1 = Linear(784, 256, bias=True)
        self.fc2 = Linear(256, 128, bias=True)
        self.fc3 = Linear(128, 10)
        self.logsoftmax = LogSoftmax()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, input):
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.logsoftmax(x, 1)

    def parameters(self):

        return [self.fc1.weights, self.fc2.weights, self.fc1.bias, self.fc2.bias]


model = Network()
# optimizer = SGD(model.parameters(), lr=0.1)
optimizer = Adam(model.parameters())
criterion = NLLLoss()

# Batching code
X_train, Y_train, X_test, Y_test = fetch_mnist()

# Normalizing data. Default MNIST is not normalized
X_train, X_test = X_train / 255-0.5, X_test / 255-0.5

epochs = 1500
losses = []

for i in range(0, epochs):
    optimizer.zero_grad()
    # Incorrect step here
    samp = np.random.randint(0, X_train.shape[0], size=(64))

    x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]), Y] = 1
    y = Tensor(y)

    out = model.forward(x)
    loss = criterion(out, y)
    loss.backward()
    losses.append(loss.value)
    optimizer.step()


plt.plot(range(0, epochs), losses)
plt.show()
