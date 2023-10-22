import math

import torch
from torch import nn

from . import sampling, utils
import todos
import pdb

# class VDenoiser(nn.Module):
#     """A v-diffusion-pytorch model wrapper for k-diffusion."""

#     def __init__(self, inner_model):
#         super().__init__()
#         self.inner_model = inner_model
#         self.sigma_data = 1.

#     def get_scalings(self, sigma):
#         c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
#         c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
#         c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
#         return c_skip, c_out, c_in

#     def sigma_to_t(self, sigma):
#         return sigma.atan() / math.pi * 2

#     def t_to_sigma(self, t):
#         return (t * math.pi / 2).tan()

#     def loss(self, input, noise, sigma, **kwargs):
#         c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
#         noised_input = input + noise * utils.append_dims(sigma, input.ndim)
#         model_output = self.inner_model(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
#         target = (input - c_skip * noised_input) / c_out
#         return (model_output - target).pow(2).flatten(1).mean(1)

#     def forward(self, input, sigma, **kwargs):
#         c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
#         return self.inner_model(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        # sigmas = tensor([ 0.029168,  0.041314,  0.050680,  ..., 14.438568, 14.526276,
        #         14.614642], device='cuda:0'), size() -- 1000
        # quantize = True
        # xxxx_refiner 3        
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())
        self.quantize = quantize

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    # def get_sigmas(self, n=None):
    #     pdb.set_trace()

    #     if n is None:
    #         return sampling.append_zero(self.sigmas.flip(0))
    #     t_max = len(self.sigmas) - 1
    #     t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
    #     return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_discrete_timestep(self, sigma):
        # self.sigmas.size() -- [1000]
        # self.log_sigmas.size() -- [1000]

        # tensor([0.149319], device='cuda:0')
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]

        return dists.abs().argmin(dim=0).view(sigma.shape)

    def sigma_to_t(self, sigma, quantize=None):
        # sigma = tensor([4.796544], device='cuda:0')
        # quantize = None
        
        quantize = self.quantize if quantize is None else quantize
        if quantize: # self.quantize -- True
            return self.sigma_to_discrete_timestep(sigma)

        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx

        pdb.set_trace()
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        # canny lora ==> pdb.set_trace(), self.quantize -- True
        t = t.float()
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t-low_idx if t.device.type == 'mps' else t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]

        # canny lora ==> pdb.set_trace()
        return log_sigma.exp()

    def predict_eps_discrete_timestep(self, input, t, **kwargs):
        if t.dtype != torch.int64 and t.dtype != torch.int32:
            t = t.round()
        sigma = self.t_to_sigma(t)
        input = input * ((utils.append_dims(sigma, input.ndim) ** 2 + 1.0) ** 0.5)
        return  (input - self(input, sigma, **kwargs)) / utils.append_dims(sigma, input.ndim)

class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)

        # alphas_cumprod.size() -- [1000]
        # quantize = True

        self.inner_model = model
        self.sigma_data = 1.
        # self.inner_model ----
        # CFGNoisePredictor(
        #   (inner_model): SDXL(
        #     (diffusion_model): UNetModel(...)))

    def get_scalings(self, sigma):
        # ==> pdb.set_trace()
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

        return c_out, c_in

    # def get_eps(self, *args, **kwargs):
    #     pdb.set_trace()
    #     return self.inner_model(*args, **kwargs)

    # def loss(self, input, noise, sigma, **kwargs):
    #     pdb.set_trace()
    #     c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
    #     noised_input = input + noise * utils.append_dims(sigma, input.ndim)
    #     eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
    #     return (eps - noise).pow(2).flatten(1).mean(1)

    # xxxx_refiner 2 ...
    def forward(self, input, sigma, **kwargs):
        # tensor [input] size: [1, 4, 75, 57], min: -3.141021, max: 3.365364, mean: -0.027016
        # tensor [sigma] size: [1], min: 0.149319, max: 0.149319, mean: 0.149319
        # kwargs.keys() -- ['cond', 'uncond', 'cond_scale', 'cond_concat', 'model_options', 'seed']

        # sigma -- tensor([0.149319], device='cuda:0')
        # ==> 
        # self.get_scalings(sigma) -- (tensor([-0.149319], device='cuda:0'), tensor([0.989035], device='cuda:0'))
        # ==> self.sigma_to_t(sigma) -- 23

        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return input + eps * c_out


# class OpenAIDenoiser(DiscreteEpsDDPMDenoiser):
#     """A wrapper for OpenAI diffusion models."""

#     def __init__(self, model, diffusion, quantize=False, has_learned_sigmas=True, device='cpu'):
#         alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
#         super().__init__(model, alphas_cumprod, quantize=quantize)
#         self.has_learned_sigmas = has_learned_sigmas

#     def get_eps(self, *args, **kwargs):
#         model_output = self.inner_model(*args, **kwargs)
#         if self.has_learned_sigmas:
#             return model_output.chunk(2, dim=1)[0]
#         return model_output


class CompVisDenoiser(DiscreteEpsDDPMDenoiser):
    """A wrapper for CompVis diffusion models."""

    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_eps(self, *args, **kwargs):
        # args is tuple: len = 2
        #     tensor [item] size: [1, 4, 75, 57], min: -3.884738, max: 3.884999, mean: 0.003339
        #     tensor [item] size: [1], min: 786.0, max: 786.0

        # kwargs is dict:
        #     list [cond] len: 1
        #     [item] value: '[tensor([[[ 0.007852, -0.466604, -0.024222,  ...,  0.190406,  0.418031,
        #           -0.296894],
        #          [-0.115181, -0.054989, -0.387905,  ..., -0.052225,  0.480097,
        #           -0.745607],
        #          [-0.150094, -0.024470, -0.249452,  ...,  0.220371, -0.056511,
        #            0.496225],
        #          ...,
        #          [-0.106456, -0.458482,  0.604304,  ...,  0.416376,  0.278748,
        #            0.165910],
        #          [-0.197682, -0.241640,  0.531077,  ...,  0.366392,  0.148722,
        #            0.185353],
        #          [ 0.023255, -0.016426,  0.578519,  ...,  0.249666,  0.172791,
        #           -0.028520]]], device='cuda:0'), {'pooled_output': tensor([[-0.087919, -1.548311, -0.571016,  ...,  0.208579, -2.006104,
        #          -0.290721]]), 'adm_encoded': tensor([[    -0.087919,     -1.548311,     -0.571016,  ...,
        #               0.000745,      0.000693,      0.000645]], device='cuda:0')}]'
        #     list [uncond] len: 1
        #     [item] value: '[tensor([[[ 0.007852, -0.466604, -0.024222,  ...,  0.190406,  0.418031,
        #           -0.296894],
        #          [-0.092446, -1.123855, -0.039772,  ...,  0.482294, -0.078357,
        #           -0.655703],
        #          [ 0.345964, -0.699447,  0.048333,  ..., -0.325103,  0.564731,
        #           -0.732478],
        #          ...,
        #          [-0.083374, -0.450125,  0.483659,  ...,  0.159920, -0.003921,
        #            0.311670],
        #          [-0.130741, -0.317075,  0.503707,  ...,  0.152807, -0.124661,
        #            0.284142],
        #          [ 0.024848, -0.290482,  0.612310,  ...,  0.152630, -0.092093,
        #            0.304129]]], device='cuda:0'), {'pooled_output': tensor([[ 0.034269,  0.089418,  0.148130,  ..., -0.289218,  0.735381,
        #          -0.745058]]), 'adm_encoded': tensor([[0.034269, 0.089418, 0.148130,  ..., 0.000310, 0.000289, 0.000269]],
        #        device='cuda:0')}]'
        #     [cond_scale] value: 7.5
        #     [cond_concat] value: None
        #     [model_options] value: {'transformer_options': {}}
        #     [seed] value: 859346782432149

        # self.inner_model.apply_model -- 
        # <bound method CFGNoisePredictor.apply_model of CFGNoisePredictor(
        #   (inner_model): SDXLRefiner(
        #     (diffusion_model): UNetModel(...)))

        return self.inner_model.apply_model(*args, **kwargs)


# class DiscreteVDDPMDenoiser(DiscreteSchedule):
#     """A wrapper for discrete schedule DDPM models that output v."""

#     def __init__(self, model, alphas_cumprod, quantize):
#         super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
#         self.inner_model = model
#         self.sigma_data = 1.

#     def get_scalings(self, sigma):
#         c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
#         c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
#         c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
#         return c_skip, c_out, c_in

#     def get_v(self, *args, **kwargs):
#         return self.inner_model(*args, **kwargs)

#     def loss(self, input, noise, sigma, **kwargs):
#         c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
#         noised_input = input + noise * utils.append_dims(sigma, input.ndim)
#         model_output = self.get_v(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
#         target = (input - c_skip * noised_input) / c_out
#         return (model_output - target).pow(2).flatten(1).mean(1)

#     def forward(self, input, sigma, **kwargs):
#         c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
#         return self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


# class CompVisVDenoiser(DiscreteVDDPMDenoiser):
#     """A wrapper for CompVis diffusion models that output v."""

#     def __init__(self, model, quantize=False, device='cpu'):
#         super().__init__(model, model.alphas_cumprod, quantize=quantize)

#     def get_v(self, x, t, cond, **kwargs):
#         return self.inner_model.apply_model(x, t, cond)
