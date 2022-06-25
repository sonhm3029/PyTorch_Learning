import numpy as np
from matplotlib import pyplot as plt

X = np.array([1,2,2,3,3,4,5,6,6,6,8,10]).reshape(-1, 1)
Y = np.array([-890, -1411, -1560, -2220,-2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157 ]).reshape(-1, 1)

plt.scatter(X, Y)
plt.xlabel("Number of hydrocarbons in molecule")
plt.ylabel("Hear release burned (kJ/mol)")
plt.show()

# NUmber of sample
N = X.shape[0]

# Thêm vào x cột 1:
X = np.concatenate((np.ones((N, 1)), X), axis = 1)
w = np.array([0., 1.]).reshape(-1, 1)
# Xây dụng model

def fx(w):
    return X.dot(w)

def MSE_loss(y, y_pred):
    return 0.5*(1/N)*(y - y_pred)**2

def grad(x, w, y):
    return (1/N)*x.T.dot(x.dot(w)-y)


# Define learning rate + epoch
lr = 0.01
epochs = 1000

# Training loop
loss = 0

# momenttum
v_init = np.array([0., 0.]).reshape(-1, 1)
gama = 0.9

for epoch in range(epochs):
    y_pred = fx(w)
    
    loss = MSE_loss(Y, y_pred)
    if loss.mean() < 1e-3:
        print("loss", loss)
        break

    if epoch % 10 == 0:
        print(f"loss {epoch} {loss.mean()}")

    loss_grad = grad(X, w, Y)

    # momentum
    v_new =  0+ lr*loss_grad
    v_init = v_new
    w -= v_new


print("loss",loss.mean())
print("w", w)

