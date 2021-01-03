# this is an example using picograd to train a fully-connected classifier on the MNIST data

import matplotlib.pyplot as plt
from picograd.Tensor import *
from picograd.nn import Linear, ReLU, LogSoftmax, Dropout
from picograd.Loss import NLLLoss
from picograd.optim import SGD, Adam
from Examples.MNIST_Helper import *
from Examples.Model_Eval import *
import time

# defining network - following PyTorch structure. One of the network architectures used in original dropout paper


class Network():
    def __init__(self):
        self.fc1 = Linear(784, 800, bias=True)
        self.fc2 = Linear(800, 800, bias=True)
        self.fc3 = Linear(800, 10)
        self.dropout = Dropout(p=0.5)
        self.logsoftmax = LogSoftmax()
        self.relu = ReLU()

    def forward(self, input):
        x = self.relu(self.fc1(input))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.logsoftmax(x, 1)

    def parameters(self):

        return [self.fc1.weights, self.fc2.weights, self.fc3.weights, self.fc1.bias, self.fc2.bias,
                self.fc3.bias]

    def eval(self):
        self.dropout.p = 0


# Initializing model, loss, optimizer
model = Network()
optimizer = Adam(model.parameters())
criterion = NLLLoss()

# getting train/test data
X_train, Y_train, X_test, Y_test = fetch_mnist()

# Normalizing data. Default MNIST is not normalized
X_train, X_test = X_train / 255-0.5, X_test / 255-0.5

# Creating train dataloader
trainloader = MNISTloader(X_train, Y_train, batch_size=64)

# Training

epochs = 1
losses = []

t0 = time.time()
for i in range(0, epochs):
    running_loss = 0

    # Need to reset iterator every epoch
    trainloader.iter = 0
    for images, labels in trainloader:

        optimizer.zero_grad()

        out = model.forward(images)

        loss = criterion(out, labels)

        loss.backward()

        running_loss += loss.value

        optimizer.step()

    if i % 2 == 0:
        print('Running Loss:', running_loss/len(trainloader))

    losses.append(running_loss/len(trainloader))
t1 = time.time()

print(
    f'Training model for {epochs} epochs. Total time to train: {t1-t0:.2f} seconds')
testloader = MNISTloader(X_test, Y_test, batch_size=64)
model.eval()
print('Test Loss:')
eval_acc(model, testloader)
print('Training Loss:')
eval_acc(model, trainloader)
