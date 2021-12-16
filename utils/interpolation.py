from __future__ import absolute_import, division

import torch
from torch import nn
import torch.nn.functional as F


def upsample_flow(flow, output_size):
    ## 扩大光流的大小
    input_size = flow.size()[2:4]
    resized_flow = F.interpolate(flow, size=output_size, mode="bilinear", align_corners=True)
    # correct scaliing of flow
    u, v = resized_flow.chunk(2, dim=1)
    u = u * float(output_size[1] / input_size[1])
    v = v * float(output_size[0] / input_size[0])

    return torch.cat([u, v], dim=1)


def get_grid(x):

    b, _, h, w = x.size()
    grid_H = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grid_V = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grids = torch.cat([grid_H, grid_V], dim=1).requires_grad_(False)
    
    return grids

def get_coordgrid(x):

    b, _, h, w = x.size()
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    ones = torch.ones_like(grid_h)
    coordgrid = torch.cat((grid_h, grid_v, ones), dim=1).requires_grad_(False)

    return coordgrid