import torch
from torch import nn, einsum
from einops import rearrange

class FastTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()

    def forward(self, x):
        return x
