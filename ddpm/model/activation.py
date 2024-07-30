import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class SiLU(nn.Module):
    r"""
    activation function SiLU/Swish: `x*sigmoid(x)`
    """
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

