from numpy import dtype
import torch, torch.nn as nn

# Các bước thực hiện

"""
1) Thiết lập input, output, design model, forward pass
2) Xây dụng loss function, optimize bai toan vs GD hoac SGD hoac Adam...
3) Training Loop:
    - forward pass: tính toán y_predicted và loss
    - backward pass: tính toán gradient
    - update weights
"""

# Sử dụng linear regression model của pytorch + SGD

# Thiết lập input, output
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

# 1) Thiet lap model linear regression:
n_samples, n_features = X.shape
input_size = output_size = n_features


LinearRegModel = nn.Linear(input_size, output_size)

# Hoac ta co the custom lai model nhu sau:

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Define layer
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

print(f"Ket qua du doan f(5) truoc khi training: {LinearRegModel(X_test).item():3f}")


# 2) Thiet lap loss function va optimizer
learning_rate = 0.03
epochs = 100

MSE_loss = nn.MSELoss()
SGD = torch.optim.SGD(LinearRegModel.parameters(), lr = learning_rate)

# 3) Training Loop

for epoch in range(epochs):

    y_predicted = LinearRegModel(X)

    loss = MSE_loss(Y, y_predicted)
    if loss <= 1e-3:
        break

    loss.backward()


    SGD.step()
    SGD.zero_grad()

    if epoch % 1 == 0:
        [w, b] = LinearRegModel.parameters()
        print(f"epoch: {epoch + 1}: w= {w[0][0].item()}, loss= {loss}")

print(f"Gia tri f(5) sau training: {LinearRegModel(X_test).item():.3f}")



