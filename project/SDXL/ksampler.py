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
import os
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

from SDXL.util import (
    make_beta_schedule,
)
from SDXL.noise import (
    CLIPEmbedNoiseAugmentation,
)

from SDXL.unet import (
    UNetModel,
)

from SDXL.controlnet import (
    ControlNet,
)


import todos
import pdb


def prepare_noise(latent_image, seed):
    generator = torch.manual_seed(seed)
    return torch.randn(latent_image.size(), dtype=latent_image.dtype, generator=generator).to(latent_image.device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def get_ancestral_step(sigma_from, sigma_to):
    sigma_up = min(sigma_to, (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def to_d(latent_noise, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (latent_noise - denoised) / append_dims(sigma, latent_noise.ndim)

# KarrasScheduler -- sigma_min, sigma_max ...
def get_karras_sigmas(n, sigma_min=0.0291675, sigma_max=14.614642, rho=7.0):
    """Constructs the noise schedule of Karras et al. (2022)."""

    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho # size() -- 13

    return torch.cat([sigmas, sigmas.new_zeros([1])])

class KSampler(nn.Module):
    def __init__(self, version="refiner_1.0"):
        super(KSampler, self).__init__()
        self.version = version

        self.scale_factor = 0.13025
        self.diffusion_model = UNetModel(version=version)
        if version == "base_1.0":
            self.lora_model = ControlNet()
        else:
            self.lora_model = nn.Identity()
        self.register_schedule()

        for param in self.parameters():
            param.requires_grad = False

    def register_schedule(self, beta_schedule="linear", timesteps=1000, linear_start=0.00085, linear_end=0.012):
        betas = make_beta_schedule(timesteps, linear_start=linear_start, linear_end=linear_end)
        # ==> torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        # betas.shape -- (1000,), range: [0.00085, 0.012]

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)

        self.register_buffer('betas', torch.tensor(betas))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod))
        self.register_buffer('sigmas', torch.tensor(sigmas))
        self.register_buffer('log_sigmas', torch.tensor(log_sigmas))

    def sigma_to_t(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]

        return dists.abs().argmin(dim=0).view(sigma.shape)

    # def t_to_sigma(self, t):
    #     t = t.float()
    #     low_idx = t.floor().long()
    #     high_idx = t.ceil().long()
    #     w = t.frac()
    #     log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]

    #     return log_sigma.exp()

    def get_scalings(self, sigma):
        # c_out = -sigma
        # c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1.0 / ((sigma ** 2 + 1.0) ** 0.5)
        return c_in

    def diffusion_predict(self, latent_noise, sigma, positive_tensor, negative_tensor, cond_scale):
        c_out = -sigma
        c_in =  append_dims(self.get_scalings(sigma), latent_noise.ndim)

        t = self.sigma_to_t(sigma)

        x2 = torch.cat((latent_noise * c_in, latent_noise * c_in), dim=0).half()
        t2 = torch.cat((t, t), dim=0).half()
        c2 = torch.cat((positive_tensor['text_encoded'], negative_tensor['text_encoded']), dim=0).half()
        y2 = torch.cat((positive_tensor['adm_encoded'], negative_tensor['adm_encoded']), dim=0).half()
        ctrl2 = {'input':[], 'middle':[], 'output': []} # control output list

        if 'lora_guide' in positive_tensor:
            if os.environ.get("SDXL_DEBUG") is not None:
                todos.debug.output_var("lora_guide", positive_tensor['lora_guide'])

            h2 = positive_tensor['lora_guide'].half()
            with torch.no_grad():
                control_output = self.lora_model(x=x2, hint=h2, timesteps=t2, context=c2, y=y2)

            weight = 1.0
            if 'lora_weight' in positive_tensor: # xxxx9999
                weight = positive_tensor['lora_weight']

            # The following come from function control_merge
            for i in range(len(control_output)):
                if i == (len(control_output) - 1):
                    key = 'middle'
                else:
                    key = 'output'
                ctrl2[key].append(control_output[i] * weight)

        with torch.no_grad():
            e2 = self.diffusion_model(x2, timesteps=t2, context=c2, y = y2, control=ctrl2)
            eps1 = e2[0:1, :, :, :]
            eps2 = e2[1:2, :, :, :]

        eps = eps2 + (eps1 - eps2) * cond_scale # uncond + (cond - uncond) * cond_scale, get_eps

        return (latent_noise + eps * c_out).to(torch.float32)

    def set_steps(self, steps, denoise=1.0):
        if denoise > 0.9999:
            sigmas = get_karras_sigmas(steps)
        else:
            denoise = max(0.01, denoise)
            new_steps = int(steps/denoise)
            sigmas = get_karras_sigmas(new_steps)
            sigmas = sigmas[-(steps + 1):]
        return sigmas

    # control_tensor['lora_guide', 'lora_weight']
    def forward(self, positive_tensor, negative_tensor, latent_image, cond_scale=7.5, steps=20, denoise=1.0, seed=-1):
        if 'lora_guide' in positive_tensor:
            pass
            # load canny lora model weight if needed ???

        B, C, H, W = latent_image.size()
        positive_tensor["adm_encoded"] = self.encode_adm(positive_tensor, H, W, positive=True)
        negative_tensor["adm_encoded"] = self.encode_adm(negative_tensor, H, W, positive=False)

        print("-" * 120)
        todos.debug.output_var("positive_tensor", positive_tensor)
        todos.debug.output_var("negative_tensor", negative_tensor)
        todos.debug.output_var("latent_image", latent_image)
        print("-" * 120)

        sigmas = self.set_steps(steps, denoise).to(latent_image.device) # steps, denois ==> sigmas

        noise = prepare_noise(latent_image, seed)
        if math.isclose(float(self.sigmas[-1]), float(sigmas[0]), rel_tol=1e-05):
            noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            noise = noise * sigmas[0]

        latent_image = self.process_latent_in(latent_image)
        latent_noise = latent_image + noise # prepare_noise(latent_image, seed) * sigmas[0]

        # forget:  steps=20, denoise=1.0, seed=-1
        # forward: latent_noise, positive_tensor, negative_tensor, cond_scale

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # https://github.com/lllyasviel/Fooocus
        # 
        # DPM family seems well-suited for XL, since XL sometimes generates overly smooth texture but DPM family sometimes 
        # generate overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.
        #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # sample = self.sample_euler_ancestral(sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale)
        # sample = self.sample_euler(sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale)
        # sample = self.sample_dpm_2_ancestral(sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale)
        sample = self.sample_dpm_2(sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale)

        latent_output = self.process_latent_out(sample) # sample

        return latent_output


    def process_latent_in(self, latent):
        return latent * self.scale_factor

    def process_latent_out(self, latent):
        return latent / self.scale_factor

    def sample_euler_ancestral(self, sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale): 
        """Ancestral sampling with Euler method steps."""

        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            denoised = self.diffusion_predict(latent_noise, sigmas[i] * s_in, positive_tensor, negative_tensor, cond_scale)

            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            d = to_d(latent_noise, sigmas[i], denoised) 

            # Euler method
            dt = sigma_down - sigmas[i]
            latent_noise = latent_noise + d * dt
            if sigmas[i + 1] > 0:
                latent_noise = latent_noise + torch.randn_like(latent_noise) * sigma_up

        return latent_noise

    def sample_euler(self, sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""

        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            sigma_hat = sigmas[i]

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            denoised = self.diffusion_predict(latent_noise, sigma_hat * s_in, positive_tensor, negative_tensor, cond_scale)

            d = to_d(latent_noise, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            latent_noise = latent_noise + d * dt

        return latent_noise


    def sample_dpm_2(self, sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale):
        """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""

        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            denoised = self.diffusion_predict(latent_noise, sigma_hat * s_in, positive_tensor, negative_tensor, cond_scale)
            d = to_d(latent_noise, sigma_hat, denoised)
            if sigmas[i + 1] == 0:
                # Euler method
                dt = sigmas[i + 1] - sigma_hat
                latent_noise = latent_noise + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
                dt_1 = sigma_mid - sigma_hat
                dt_2 = sigmas[i + 1] - sigma_hat
                x_2 = latent_noise + d * dt_1

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # model_forward
                #
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                denoised_2 = self.diffusion_predict(x_2, sigma_mid * s_in, positive_tensor, negative_tensor, cond_scale)
                d_2 = to_d(x_2, sigma_mid, denoised_2)
                latent_noise = latent_noise + d_2 * dt_2
        return latent_noise


    def sample_dpm_2_ancestral(self, sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale):
        """Ancestral sampling with DPM-Solver second-order steps."""

        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            denoised = self.diffusion_predict(latent_noise, sigmas[i] * s_in, positive_tensor, negative_tensor, cond_scale)

            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])

            d = to_d(latent_noise, sigmas[i], denoised)

            if sigma_down == 0:
                # Euler method
                dt = sigma_down - sigmas[i]
                latent_noise = latent_noise + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
                dt_1 = sigma_mid - sigmas[i]
                dt_2 = sigma_down - sigmas[i]
                x_2 = latent_noise + d * dt_1
                
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # model_forward
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                denoised_2 = self.diffusion_predict(x_2, sigma_mid * s_in, positive_tensor, negative_tensor, cond_scale)

                d_2 = to_d(x_2, sigma_mid, denoised_2)
                latent_noise = latent_noise + d_2 * dt_2
                latent_noise = latent_noise + torch.randn_like(latent_noise) * sigma_up
        return latent_noise


    def encode_adm(self, cond, H, W, positive=True):
        pass
        return cond


if __name__ == "__main__":
    model = KSampler(version="refiner_1.0")
    model = model.eval()
    # model = torch.jit.script(model)
    model = model.cuda()

    # NO load weights, just test process ... !!!!!!!!!!!!!!!!!!!!!!!!
    positive_tensor = {
        "text_encoded" : torch.randn(1, 77, 1280).cuda(),
        "pool_encoded" : torch.randn(1, 1280).cuda(),
    }

    negative_tensor = {
        "text_encoded" : torch.randn(1, 77, 1280).cuda(),
        "pool_encoded" : torch.randn(1, 1280).cuda(),
    }

    latent_image = torch.randn(1, 4, 75, 57).cuda()
    cond_scale = 7.5
    steps = 200
    denoise = 0.2
    seed = -1

    with torch.no_grad():
        sample_out = model(positive_tensor, negative_tensor, latent_image, cond_scale, steps, denoise, seed)

    todos.debug.output_var("sample_out", sample_out)