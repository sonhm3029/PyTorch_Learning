# Gradient descent với bài toán sử dụng linear regression:
# import numpy as np
import torch, torch.nn as nn 

# Cac buoc co ban cua bai toan NN :
"""
1) Design model ( input, ouput size, forward pass)
2) Xay dung loss function, optimize bai toan vs GD hoac SGD
3) Training loop
    - forward pass: xac dinh dau ra du doan va loss
    - backward pass: tinh toan np.gradient
    - update weights
"""

# ham ban dau: f = w*x voi w = 2
# Tap du lieu training:
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

# Chon gia tri khoi tao weight:
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model output:
def forward(x):
    return w*x

#Loss
MSE_loss = nn.MSELoss()

#gradient cua loss function:
# loss = 1/N * (y_predicted - y)**2
#      = 1/N * (w*x - y)**2
# dloss/dw = 1/N * 2*x*(w*x - y)
def gradient(x, y, y_predicted):
    return (2*x*(y_predicted - y)).mean()

print(f"Du doan truoc khi training: f(5) = {forward(5):.3f}")

# Training loop
learning_rate = 0.01
epochs = 20

# Su dung SGD:
SGD = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(epochs):
    y_pred = forward(X)

    #loss
    loss = MSE_loss(Y, y_pred)

    # Thoat vong lap khi co loss du nho
    if loss < 1e-3:
        break

    # Auto run backward to calculate gradient
    loss.backward()

    # Update weight
    SGD.step()

    # In ra de theo doi weight
    if epoch % 2 ==0:
        print(f"epoch {epoch + 1}: w = {w:3f}, loss = {loss:3f}")
    
    # Empty grad
    SGD.zero_grad()


print(f"Gia tri du doan sau khi training: f(5) = {forward(5):.3f}")
