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
input_size = 784 #28*28
hidden_size_1 = 392
hidden_size_2 = 196
hidden_size_3 = 98
hidden_size_4 = 49
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 1e-3

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
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size_1)
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.l3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.l4 = nn.Linear(hidden_size_3, hidden_size_4)
        self.relu = nn.ReLU()
        self.l = nn.Linear(hidden_size_4, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l(out)
        
        return out
    
model = NeuralNet(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, num_classes)

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