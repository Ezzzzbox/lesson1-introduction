import torch
import numpy as np
from math import sin

# 2.1
x = torch.tensor([0.0, 1, 2, 3], requires_grad=True)
y = torch.tensor([0.0, 1, 2, 3], requires_grad=True)
z = torch.tensor([0.0, 1, 2, 3], requires_grad=True)
f = x**2 + y**2 + z**2 + 2 * x * y * z
# print(f)
# 0 index = 0 + 0 + 0 + 0 * 2 = 0
# 1 index = 1 + 1 + 1 + 1 * 2 = 5
# 2 index = 4 + 4 + 4 + 2 * 2 * 2 * 2 = 28
# 3 index = 9 + 9 + 9 + 2 * 3 * 3 * 3 = 81

m = f.sum()
m.backward()
print("0 4 12 24")
print(x.grad)
print(y.grad)
print(z.grad)

# 2.2
w = torch.tensor([1.0, 1])
X = torch.tensor([[1.0, 2], [3, 4]])
b = torch.tensor([1.0, 1])
y_true = torch.tensor([800.0, 800])
y_pred = X @ w + b
mse = torch.mean((y_true - y_pred) ** 2)
print(mse)

# 2.3
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
f = torch.sin(x**2 + 1)
res = f.sum()
res.backward()
print(x.grad)
