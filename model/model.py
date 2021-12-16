from re import L
from numpy.core.defchararray import encode
import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.activation import ReLU


def build_grid(resolution):
    # [256, 832]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(0., 1., num = res) for res in resolution]
    ## *parameter接收多个参数并放入一个元组中
    ## **parameter接收一个字典
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    # range[0, 1]
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    ## 正向和反向的grid
    ## [0, 0] + [100, 100]
    ## [0. 1] + [100, 99]
    return torch.tensor(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """
        Builds the soft position embedding layer.
        Args:
            hidden_size: Size of input feature dimension.
            resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        # input features -> output features
        self.proj = nn.Linear(4, hidden_size)
        self.grid = build_grid(resolution)
    
    def forward(self, inputs):
        ## 4维映射至hidden_channel
        return inputs + self.proj(self.grid)


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid"""
    # slots has (b, 2, D)
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid


def spatial_flatten(x):
    return torch.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3):
    unstacked = einops.rearrange(x, '(b s) c h w -> b s c h w', b=batch_size)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
    return channels, masks


class SlotAttention(nn.Module):
    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """
        slot attention 
        """
        super(SlotAttention, self).__init__()
        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # parameters for gaussian init
        self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # 用高斯来初始化参数
        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # 线性映射
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # gru
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)
        hidden_dim = max(encoder_dims, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )


    def forward(self, inputs, num_slots=None):
        # input size [batch_size, num_inputs, inputs_size]
        inputs = self.norm_input(inputs)
        # batchsize * HW * Dimension
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        # INitialize the slots 
        b, n ,d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)
        
        # learnable slots initialization
        # batch_size
        # (B, NS, ENCODERDIM)
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        for _ in range(self.iters):
            slots_prev = slots
            # b 2 d
            slots = self.norm_slots(slots)

            # attention
            q = self.project_q(slots)
            # [batch, 2, dimension]
            dots = torch.einsum('bid, bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True) # weighted mean
            
            # (b, 2, d)
            updates = torch.einsum('bjd, bij->bid', v, attn)
            # slots update
            # H1 = (b * 2)
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            # update
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, in_out_channels=3, iters=5):
        super(SlotAttentionAutoEncoder, self).__init__()

        self.iters = iters
        self.resolution = resolution
        self.num_slots = num_slots
        self.in_out_channels = in_out_channels

        # 2次下采样
        self.encoder_arch = [64, 'MP', 128, 'MP', 256]
        self.encoder_dims = self.encoder_arch[-1]
        self.encoder_cnn, ratio = self.make_encoder(self.in_out_channels, self.encoder_arch)
        ## encoder & position embedding
        self.encoder_end_size = (int(resolution[0] / ratio), int(resolution[1] / ratio))
        self.encoder_pos = SoftPositionEmbed(self.encoder_dims, self.encoder_end_size)
        ## decoder & position embedding
        self.decoder_initial_size = (int(resolution[0] / 8), int(resolution[1] / 8))
        self.decoder_pos = SoftPositionEmbed(self.encoder_dims, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(self.encoder_dims)

        ## 定义一个mlp过程
        ## input = output
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.ReLU(inplace = True),
            nn.Linear(self.encoder_dims, self.encoder_dims)
        )

        self.slot_attention = SlotAttention(
            iters = self.iters,
            num_slots = self.num_slots,
            encoder_dims= self.encoder_dims,
            hidden_dim = self.encoder_dims
        )


        ## return image & mask
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_dims, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_out_channels + 1, kernel_size=5, padding=2, stride=1)
        )


    def make_encoder(self, in_channels, encoder_arch):
        layers = []
        down_factor = 0
        for v in encoder_arch:
            if v == 'MP':
                layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                # 下采样数++
                down_factor += 1
            else: 
                conv1 = nn.Conv2d(in_channels, v, kernel_size=5, padding=2)
                conv2 = nn.Conv2d(v, v, kernel_size=5, padding=2)
                layers += [conv1, nn.InstanceNorm2d(v ,affine=True)]
                in_channels = v
            
        return nn.Sequential(*layers), 2 ** down_factor


    def forward(self, image):
        # image has shape (b, c, h, w)
        # cnn feature extractor
        x = self.encoder_cnn(image)
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.encoder_pos(x)  # position embedding
        x = spatial_flatten(x)
        # feature embedding + position embedding
        # + mlp => input to slot attention
        x = self.mlp(self.layer_norm(x))
        slots = self.slot_attention(x)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        # (b*numslot, h, w, dimension)
        x = self.decoder_pos(x)
        x = einops.rearrange(x, 'b_n h w c -> b_n c h w')
        x = self.decoder_cnn(x)

        # x shape(b*numslots, c+1, height, width)

        recons, masks = unstack_and_split(x, batch_size=image.shape[0], num_channels=self.in_out_channels)
        # recons (b, numslots, c, h, w)
        # masks (b, numslots, 1, h, w)

        # normalize masks over slots => softmax
        masks = torch.softmax(masks, axis=1)

        recon_combined = torch.sum(recons * masks, axis=1)

        return recon_combined, recons, masks, slots



""" if __name__ == "__main__":

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # q = torch.randn((4, 2, 6))
    model = SlotAttentionAutoEncoder(
        resolution=(256, 832),
        num_slots=2,
        in_out_channels=3,
        iters=5
    )
    print(model) """

