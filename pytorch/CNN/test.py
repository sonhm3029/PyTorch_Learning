import os
from re import X
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters

num_epochs = 5
batch_size = 4
learning_rate = 0.001

# Data loading in
transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10
train_dataset = datasets.CIFAR10(root="././data", train=True, 
                                 download=False, transform=transform )

test_dataset = datasets.CIFAR10(root="././data", train=False, 
                                 download=False, transform=transform )

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                          shuffle=False)

print(train_dataset.data.shape)
print(train_dataset.data[0])


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Show data
def imshow(images):
    images = images/2 + 0.5
    npimg = images.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
# Get some example
examples = iter(train_loader)
images, labels = examples.next()
imshow(torchvision.utils.make_grid(images))

# define neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
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

# training model

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        
        # forward 
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward an optimie
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(i + 1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            
print('Finished Training')      


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]  
    n_class_samples = [0 for i in range(10)]  
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels)
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if( label == pred):
                n_class_correct[label] +=1
            n_class_samples[label] +=1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')        