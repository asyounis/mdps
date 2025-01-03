# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Tuple

import numpy as np
import torch
from torch.fft import irfftn, rfftn
from torch.nn.functional import grid_sample, log_softmax, pad

from utils.general import make_grid, rotmat2d



class TemplateSampler(torch.nn.Module):
    def __init__(self, grid_xz_bev, ppm, num_rotations, optimize=True):
        super().__init__()

        grid_xz_bev = grid_xz_bev.cpu()

        Δ = 1 / ppm
        h, w = grid_xz_bev.shape[:2]
        ksize = max(w, h * 2 + 1)
        radius = ksize * Δ
        grid_xy = make_grid(radius, radius, step_x=Δ, step_y=Δ, orig_y=(Δ - radius) / 2, orig_x=(Δ - radius) / 2, y_up=True,)

        if optimize:
            assert (num_rotations % 4) == 0
            angles = torch.arange(0, 90, 90 / (num_rotations // 4), device=grid_xz_bev.device)
        else:
            angles = torch.arange(0, 360, 360 / num_rotations, device=grid_xz_bev.device)

        # Create the grid in terms of xy and rotations
        rotmats = rotmat2d(angles / 180 * np.pi)
        grid_xy_rot = torch.einsum("...nij,...hwj->...nhwi", rotmats.cpu(), grid_xy.cpu())

        # get the grid in terms of image dims
        grid_ij_rot = (grid_xy_rot - grid_xz_bev[..., :1, :1, :]) * grid_xy.new_tensor([1, -1])
        grid_ij_rot = grid_ij_rot / Δ
        grid_norm = (grid_ij_rot + 0.5) / grid_ij_rot.new_tensor([w, h]) * 2 - 1

        self.optimize = optimize
        self.num_rots = num_rotations
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer("grid_norm", grid_norm, persistent=False)

    def forward(self, image_bev):


        grid = self.grid_norm
        b, c = image_bev.shape[:2]
        n, h, w = grid.shape[:3]
        grid = grid[None].repeat_interleave(b, 0).reshape(b * n, h, w, 2)

        image = image_bev[:, None].repeat_interleave(n, 1).reshape(b * n, *image_bev.shape[1:])

        kernels = grid_sample(image, grid.to(image.dtype), align_corners=False)
        kernels = torch.reshape(kernels, (b, n, c, h, w))


        # we have computed only the first quadrant
        if self.optimize:  
            kernels_quad234 = [torch.rot90(kernels, -i, (-2, -1)) for i in (1, 2, 3)]
            kernels = torch.cat([kernels] + kernels_quad234, 1)

        return kernels




# #@torch.compile(mode="reduce-overhead")
def conv2d_fft_batchwise(signal, kernel, padding="same", padding_mode="constant"):


    if padding == "same":
        padding = [i // 2 for i in kernel.shape[-2:]]

    padding_signal = [p for p in padding[::-1] for _ in range(2)]
    signal = pad(signal, padding_signal, mode=padding_mode)
    assert signal.size(-1) % 2 == 0

    padding_kernel = [pad for i in [1, 2] for pad in [0, signal.size(-i) - kernel.size(-i)]]
    kernel_padded = pad(kernel, padding_kernel)

    signal_fr = rfftn(signal, dim=(-1, -2))
    kernel_fr = rfftn(kernel_padded, dim=(-1, -2))


    kernel_fr.imag *= -1  # flip the kernel
    output_fr = torch.einsum("bc...,bdc...->bd...", signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=(-1, -2))

    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [slice(0, (signal.size(i) - kernel.size(i) + 1)) for i in [-2, -1]]
    output = output[crop_slices].contiguous()

    return output



