import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms, utils

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = MNIST(root='/MNIST', train=True,
                   download=False, transform=transform)

test_data = MNIST(root='/MNIST', train=False,
                  download=True, transform=transform)


trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

testloader = DataLoader(test_data, batch_size=64, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        input = input.view(-1, 28*28)

        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


criterion = nn.NLLLoss()

model = Network()
model.cuda()

optimizer = torch.optim.Adam(model.parameters())
epochs = 10

for e in range(1, epochs+1):

    running_loss = 0

    for images, labels in trainloader:

        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(running_loss/len(trainloader))


# Evaluate model on test data
accuracy = 0
# Turning off gradient tracking
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        logps = model.forward(inputs)

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


print('Test Accuracy:', 100*accuracy/len(testloader), '%')
