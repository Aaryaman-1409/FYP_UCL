import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = 'cuda'
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=3, norm=True):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)


class NextNet(nn.Module):
    """
    A backbone model comprised of a chain of ConvNext blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """

    def __init__(self, in_channels, out_channels, depth=16, filter_size=64):
        super().__init__()

        dims = filter_size * depth

        time_dim = dims
        emb_dim = time_dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims, emb_dim=emb_dim, norm=False))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims, dims, emb_dim=emb_dim, norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims, dims, emb_dim=emb_dim, norm=True))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims, out_channels, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, x, t):
        embedding = self.time_encoder(t)
        print(embedding.shape)
        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            print(x.shape)
            print(residuals[-1].shape)
            print(len(residuals))
            print("")
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, embedding)

        return self.final_conv(x)
