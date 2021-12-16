import torch
import einops

def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid"""
    # slots has (b, 2, D)
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    print(slots.size())
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid

if __name__ == '__main__':
    slot = torch.randn((4, 2, 64))
    grid = spatial_broadcast(slot, (128, 256))
    print(grid)