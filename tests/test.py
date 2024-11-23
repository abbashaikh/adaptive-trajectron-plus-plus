import torch

a = torch.ones((5, 4), dtype=torch.bool)
print(a)
b = torch.triu(a, diagonal=0)
print(b)
print(b[3])