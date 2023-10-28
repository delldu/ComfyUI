from ..diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from ..diffusionmodules.openaimodel import Timestep
import torch
import todos
import pdb

class CLIPEmbedNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    def __init__(self, *args, clip_stats_path=None, timestep_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_stats_path is None: # True for refiner model
            clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        else:
            clip_mean, clip_std = torch.load(clip_stats_path, map_location="cpu")
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)
        self.time_embed = Timestep(timestep_dim) # timestep_dim == 1280 for refiner model

    def scale(self, x):
        # re-normalize to centered mean and unit variance
        x = (x - self.data_mean) * 1. / self.data_std
        return x

    def unscale(self, x):
        # back to original data stats
        x = (x * self.data_std) + self.data_mean
        return x

    def forward(self, x, noise_level=None):
        # todos.debug.output_var("x", x)
        # todos.debug.output_var("noise_level", noise_level)

        # tensor [x] size: [1280], min: -5.467596, max: 5.339845, mean: -0.032329
        # tensor [noise_level] size: [1], min: 250.0, max: 250.0 -- 0.25 why ?
        # tensor [x] size: [1280], min: -5.467596, max: 5.339845, mean: -0.032329
        # tensor [noise_level] size: [1], min: 749.0, max: 749.0 -- 0.75 why ?
        # revision ==> pdb.set_trace()
        
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        x = self.scale(x)
        z = self.q_sample(x, noise_level)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level
