from email.mime import image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
    MNIST
    DataLoader, Transformation
    Multilayer Neural Network, activate function
    Loss and Optimizer
    Training Loop (batch training)
    Model evaluation
    GPU support
"""


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root="././data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root="././data",
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)


for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
    
plt.show()  


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_totals_step = len(train_loader)

for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28 shape
        # 100, 784
        # reshape images
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        # Forward 
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # update parameters
        optimizer.step()
        
        if( idx + 1) % 100 == 0:
            print(f"epoch {epoch + 1} / {num_epochs}, step {idx+1}/{n_totals_step}, loss={loss.item():.4f}")
            
            

# Test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # reshape images
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # value, index
        _, preds = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct = (preds == labels).sum().item()
        
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')