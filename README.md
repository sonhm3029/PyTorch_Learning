# PyTorch_Learning


## 1. Các phép toán trên mảng tương tự như numpy

- PyTorch sử dụng `torch.tensor` thay cho `numpy.array`

```Python
import torch

ex_arr = torch.tensor([1,2,3,4])

```

- Để thực hiện phép nhân đại số 2 ma trận 2 chiều:

```Python
a = torch.tensor([1,2,3])
b = torh.tensor([
    [1,2],
    [3,4],
    [5,6]
])
# Thực hiện nhân 2 ma trận 1x3 vs 3x2
c = a @ b
# Hoặc
c = torch.matmul(a,b)
```

Các hàm tương tự numpy:

- `torch.zeros`: Creates a tensor filled with zeros

- `torch.ones`: Creates a tensor filled with ones

- `torch.rand`: Creates a tensor with random values uniformly sampled between 0 and 1

- `torch.randn`: Creates a tensor with random values sampled from a normal distribution with mean 0 and variance 1

- `torch.arange`: Creates a tensor containing the values 

- `torch.Tensor` (input list): Creates a tensor from the list elements you provide

**Chuyển đổi tensor to numpy và ngược lại:**

1. Numpy to tensor:

```Python
a = np.array([1,2,3,4])
b = torch.from_numpy(a)

```

2. Tensor to numpy

```Python
a = torch.tensor([1,2,3,4])
b = a.numpy()

```

## 2. Dynamic Computation Graph and 

Pytorch tự động tính toán đạo hàm của function mà ta dùng.

**Ví dụ:**

Tính đạo hàm

![](./images/1.png)

Ta có thể dùng Pytorch để tự động tính đạo hàm như sau:

Đầu tiên cần chỉ ra biến nào cần tính đạo hàm, ở đây là x:

```Python
x = torch.arange(3, dtype=torch.float32, requires_grad=True) # Only float tensors can have gradients
print("X", x)
```

Sau đó thực hiện tính toán y như bình thường:

```Python
a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
print("Y", y)
```

Tính đạo hàm của y theo x qua backpropagation

```Python
y.backward()
print(x.grad)
```

Kết quả:

![](./images/2.png)

Có thể thử lại kết quả bằng tay

## 3. Code để chạy bằng GPU nếu có:

Đầu tiên cần xem xem máy có GPU hay không:

```Python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
```

Sau đó thực hiện push code lên device đã khai báo( đây là device GPU nếu có, nếu không thì nó là CPU)

```Python
x = torch.zeros(2, 3)
x = x.to(device)
print("X", x)
```

Kết quả:

![](./images/3.png)

Trong trường hợp push code vào GPU ( máy có GPU ) thì kết quả sẽ có thể `device='cuda:0'` như sau:

![](./images/4.png)




## 4. Tensor

