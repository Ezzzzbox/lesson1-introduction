import torch
import numpy as np

# 1.1
tensor = torch.rand((3, 4))
tensor = torch.zeros((2, 3, 3))
tensor = torch.ones((5, 5))
tensor = torch.arange(0, 16).reshape(4, 4)

# 1.2
tensorA = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [4, 4, 4, 4]])
tensorA_T = tensorA.T
tensorB = torch.tensor([[1, 2, 2], [5, 4, 2], [9, 1, 9], [0, 1, 2]])
res = tensorA * tensorB.T
sum = tensorB.sum()

# 1.3
tensor = torch.rand((5, 5, 5))
# print(tensor)
# print(tensor[0, 0, :])  # first line
# print(tensor[-1, :, -1])  # last column
# print(tensor[2, 1:3, 1:3])
# print(tensor[::2, ::2, ::2])

# 1.4
big_tensor = torch.arange(0, 24)
# print(big_tensor.reshape(2, 12))
# print(big_tensor.reshape(3, 8))
# print(big_tensor.reshape(4, 6))
# print(big_tensor.reshape(2, 3, 4))
# print(big_tensor.reshape(2, 2, 2, 3))
