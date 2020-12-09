import matplotlib.pyplot as plt
from Tensor import *

from nn import Linear, ReLU, LogSoftmax, Sigmoid, Tanh

from Loss import NLLLoss

from optim import SGD, Adam

from MNIST_Helper import *


class Network():

    def __init__(self):
        self.fc1 = Linear(784, 256, bias=True)
        self.fc2 = Linear(256, 10, bias=True)
        self.logsoftmax = LogSoftmax()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, input):
        x = self.sigmoid(self.fc1(input))
        x = self.fc2(x)
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

# Creating dataloader
loader = MNISTloader(X_train, Y_train, batch_size=64)

epochs = 10
losses = []

for i in range(0, epochs):
    running_loss = 0

    # Need to reset iterator every epoch
    loader.iter = 0
    for images, labels in loader:

        optimizer.zero_grad()

        out = model.forward(images)

        loss = criterion(out, labels)

        loss.backward()

        running_loss += loss.value

        optimizer.step()

    print('Running Loss:', running_loss/len(loader))
    losses.append(running_loss/len(loader))

plt.plot(range(0, epochs), losses)
plt.show()
