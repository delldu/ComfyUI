"""SDXL 1.0 Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#
import torch
import torch.nn as nn
import numpy as np
from functools import partial

from SDXL.util import (
    Timestep,
    make_beta_schedule,
    count_model_params,
)

from typing import Optional, Tuple

import pdb


def extract_into_tensor(a, t):
    # pp a.size() -- torch.Size([1000])
    # t = tensor([0], device='cuda:0')
    # x_shape = torch.Size([1, 1280])

    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, 1)


class CLIPEmbedNoiseAugmentation(nn.Module):
    def __init__(self, max_noise_level=1000, timestep_dim=1280):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.time_embed = Timestep(timestep_dim)

        self.register_schedule()
        clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)

        for param in self.parameters():
            param.requires_grad = False
        self.half().eval()
        count_model_params(self)

    def register_schedule(self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        betas = make_beta_schedule(timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))

    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t) * noise
        )

    def scale(self, x):
        x = (x - self.data_mean) * 1.0 / self.data_std
        return x

    def unscale(self, x):
        x = (x * self.data_std) + self.data_mean
        return x

    def forward(self, x, noise_level) -> Tuple[torch.Tensor, torch.Tensor]:
        # x.size() -- [1280]
        # t = tensor([10])

        x = self.scale(x)
        z = self.q_sample(x, noise_level)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level


if __name__ == "__main__":
    model = CLIPEmbedNoiseAugmentation()
    class_name = model.__class__.__name__
    model = torch.jit.script(model)
    print(f"torch.jit.script({class_name}) OK !")
