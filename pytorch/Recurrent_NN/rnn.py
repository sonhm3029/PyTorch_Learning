import torch
import torch.nn as nn
import torch.functional as f

from utils import ALL_LETTERS, N_LETTERS
from utils import letter_to_tensor, load_data, random_training_example

import matplotlib.pyplot as plt



class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        combine_dim = input_dim + hidden_dim
        self.hidden_size = hidden_dim
        self.i2o = nn.Linear(combine_dim, output_dim)
        self.i2h = nn.Linear(combine_dim, hidden_dim)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, hidden_tensor):
        combine = torch.cat((input_tensor, hidden_tensor), 1)
        output = self.i2o(combine)
        output = self.softmax(output)
        
        hidden = self.i2h(combine)
        
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
def category_from_output(output, all_categories):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


    
if __name__ == "__main__":
    category_lines, all_categories = load_data()
    n_categoris = len(all_categories)
    n_hidden = 128
    
    rnn = RNN(N_LETTERS, n_hidden, n_categoris)
    
    # Training
    criterion = nn.NLLLoss()
    lr = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
    
    
    def train(line_tensor, category_tensor):
        hidden_tensor = rnn.init_hidden()
        for x_i in line_tensor:
            output, hidden_tensor = rnn(x_i, hidden_tensor)
        loss = criterion(output, category_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return output, loss.item()
    
    
    loss_his = []
    current_loss = 0
    epochs = 100000
    plot_steps = 1000
    print_steps = 5000
    
    
    # Training
    for epoch in range(epochs):
        category, line,\
        category_tensor, line_tensor = random_training_example(
            category_lines, all_categories)
        
        output, loss = train(line_tensor, category_tensor)
        current_loss += loss
        
        if (epoch + 1)%plot_steps == 0:
            loss_his.append(current_loss / plot_steps)
            current_loss = 0
        if (epoch+1) % print_steps == 0:
            guess = category_from_output(output, all_categories)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f} - {line} / {guess} {correct}")
            
    
    # Save model
    torch.save(rnn.state_dict(), "C:/Users/hoang/OneDrive/Desktop/PyTorch_Learning/pytorch/Recurrent_NN/data/weights.pth")
    plt.figure()
    plt.plot(loss_his)
    plt.show()
    
    
    
    
    
    