import torch
import numpy as np

# Collocation points
t_colloc = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)
t_total = torch.tensor(1.0, requires_grad=True)

# 2d
ao_2d = np.array([[-0.5, -1.0, 0.5], [-0.2, 0.4, 1.0], [0.8, 0.3, 0.5]])
x0_2d = torch.tensor([[-1.0, -1.0]])
xN_2d = torch.tensor([[1.0, 1.0]])

# 3d
ao_3d = np.array([[-0.5, -1.0, 0.0, 0.5], [-0.2, 0.4, 0.0, 1.0], [0.8, 0.3, 0.0, 0.5]])
x0_3d = torch.tensor([[-1.0, -1.0, 0.0]])
xN_3d = torch.tensor([[1.0, 1.0, 0.0]])
v0_3d = torch.tensor([[1.0, 1.0, 1.0]])
vN_3d = torch.tensor([[1.0, 1.0, -1.0]])
