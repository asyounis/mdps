
# Python Imports

# Module Imports
import torch
import numpy as np

# Project Imports

#@torch.compile(mode="reduce-overhead")
def make_grid(
    w,
    h,
    step_x = 1.0,
    step_y = 1.0,
    orig_x = 0,
    orig_y = 0,
    y_up = False,
    device = None):

    x = torch.arange(orig_x, w + orig_x, step_x, device=device)
    y = torch.arange(orig_y, h + orig_y, step_y, device=device)
    x, y = torch.meshgrid([x, y], indexing="xy")
    
    if y_up:
        y = y.flip(-2)
    

    grid = torch.stack((x, y), -1)
    return grid

#@torch.compile(mode="reduce-overhead")
def rotmat2d(angle):
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([c, -s, s, c], -1).reshape(angle.shape + (2, 2))
    return R


#@torch.compile(mode="reduce-overhead")
def sample_xyr(volume, xy_grid, angle_grid, nearest_for_inf=False):

    # (B, C, H, W, N) to (B, C, H, W, N+1)
    volume_padded = torch.nn.functional.pad(volume, [0, 1, 0, 0, 0, 0], mode="circular")

    size = xy_grid.new_tensor(volume.shape[-3:-1][::-1])
    xy_norm = xy_grid / (size - 1)  # align_corners=True
    
    # Original methods assumed degrees
    # angle_norm = (angle_grid / 360) % 1 

    # We will assume radians
    angle_norm = angle_grid % (2*np.pi)
    angle_norm = angle_norm / (2*np.pi)

    grid = torch.concat([angle_norm.unsqueeze(-1), xy_norm], -1)
    grid_norm = grid * 2 - 1

    valid = torch.all((grid_norm >= -1) & (grid_norm <= 1), -1)
    value = torch.nn.functional.grid_sample(volume_padded, grid_norm, align_corners=True, mode="bilinear")

    # if one of the values used for linear interpolation is infinite,
    # we fallback to nearest to avoid propagating inf
    if nearest_for_inf:
        value_nearest = torch.nn.functional.grid_sample( volume_padded, grid_norm, align_corners=True, mode="nearest")
        value = torch.where(~torch.isfinite(value) & valid, value_nearest, value)

    return value, valid




#@torch.compile(mode="reduce-overhead")
def nll_loss_xyr(log_probs, xy, angle):
    log_prob, _ = sample_xyr(log_probs.unsqueeze(1), xy[:, None, None, None], angle[:, None, None, None])
    nll = -log_prob.reshape(-1)  # remove C,H,W,N
    return nll


