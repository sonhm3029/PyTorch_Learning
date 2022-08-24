import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# dataset and transforms
# mean for each channels
mean = (0.5, 0.5, 0.5)
# standard deviation
std = (0.5, 0.5, 0.5)
# For pytorch, Normalize is Standardlization
composedTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root="././data",
                                             train=True,
                                             download=True,
                                             transform=composedTransform)

test_dataset = datasets.CIFAR10(root="././data",
                                             train=False,
                                             download=True,
                                             transform=composedTransform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# Implement convolution neural network

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # after conv1, from 3x32x32 to 6x28x28
        # after pool1, from 6x28x28 to 6x14x14
        # after conv2, from 6x14x14 to 16x10x10
        # after pool2, from 16x10x10 to 16x5x5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        
    

model = ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
n_totals_step = len(train_loader)

for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        # original shape : [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # update parameters
        optimizer.step()
        
        if ( idx + 1) % 2000 == 0:
            print(f"epoch {epoch + 1} / {num_epochs}, step {idx+1}/{n_totals_step}, loss={loss.item():.4f}")
            
print("Finished Training")

# Test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        ouputs = model(images)
        
        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples +=labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if( label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] +=1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network = {acc}%')
    
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]} = {acc}%')
        
        
