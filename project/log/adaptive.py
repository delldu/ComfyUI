import torch
import torch.nn.functional as F
from typing import Tuple

def gaussian(window_size: int, sigma):
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel1d(kernel_size: int, sigma):
    return gaussian(kernel_size, sigma)


def get_gaussian_kernel2d(kernel_size, sigma):
    sigma = torch.Tensor([[sigma, sigma]])

    ksize_y, ksize_x = kernel_size
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def _bilateral_blur(input, guidance, kernel_size: Tuple[int, int], sigma_color: float, sigma_space: float):
    ky, kx = kernel_size
    pad_y, pad_x = (ky - 1)//2, (kx - 1)//2

    padded_input = F.pad(input, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    padded_guidance = F.pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
    unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    color_distance_sq = diff.abs().sum(1, keepdim=True).square() # "l1"
    # color_distance_sq = diff.square().sum(1, keepdim=True) # "l2"

    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)
    space_kernel = space_kernel.to(input.device)

    kernel = space_kernel * color_kernel

    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def adaptive_filter(x, g=None):
    if g is None:
        g = x
    s, m = torch.std_mean(g, dim=(1, 2, 3), keepdim=True)
    s = s + 1e-5
    guidance = (g - m) / s
    y = _bilateral_blur(x, guidance, kernel_size=(13, 13), sigma_color=3.0, sigma_space=3.0)

    return y


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    g = torch.randn(1, 3, 128, 128)
    print(adaptive_filter(x, g).size())

    print("-" * 120)

    x = torch.randn(2, 3, 128, 128).cuda()
    g = torch.randn(2, 3, 128, 128).cuda()
    print(adaptive_filter(x, g).size())

