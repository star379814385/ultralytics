from torch import nn
import torch
class GetByIndex(nn.Module):
    def __init__(self, c1, c2, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]