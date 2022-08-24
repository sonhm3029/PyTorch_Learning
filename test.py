import torch

x = torch.arange(3, dtype=torch.float32, requires_grad=True)
print(x)
y = x+3
z = y*2
z.backward(torch.Tensor([1.0,1.0,1.0]))
print(x.grad)