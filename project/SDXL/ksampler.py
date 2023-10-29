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

import todos
import pdb


from SDXL.unet import (
    Timestep,
    UNetModel,
)

# class UNetModel(nn.Module):
#     def __init__(self, version="refiner_1.0"):
#         super(UNetModel, self).__init__()
#         self.version = version

#     def forward(self, x, timesteps=None, context=None, y=None, control=None):
#         # input ----
#         # tensor [x] size: [2, 4, 75, 57], min: -2.976877, max: 3.222236, mean: -0.02592
#         # tensor [timesteps] size: [2], min: 0.0, max: 0.0, mean: 0.0
#         # tensor [context] size: [2, 77, 1280], min: -66.1875, max: 18.375, mean: 0.032318
#         # tensor [c_adm] size: [2, 2560], min: -3.958984, max: 3.410156, mean: 0.195679
#         # [control] value: None
#         # output ----
#         # tensor [output] size: [2, 4, 75, 57], min: -3.130859, max: 3.892578, mean: -0.005257        
#         import time

#         time.sleep(1.0)
#         return x


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
        # self.embedder = Timestep(256)
        # self.noise_augmentor = CLIPEmbedNoiseAugmentation()

        self.register_schedule(beta_schedule="linear", timesteps=1000, linear_start=0.00085, linear_end=0.012)

        for param in self.parameters():
            param.requires_grad = False

    def register_schedule(self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        # beta_schedule = 'linear'
        # timesteps = 1000
        # linear_start = 0.00085
        # linear_end = 0.012

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

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))
        self.register_buffer('log_sigmas', torch.tensor(log_sigmas, dtype=torch.float32))

    def sigma_to_t(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]

        return dists.abs().argmin(dim=0).view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]

        return log_sigma.exp()

    def get_scalings(self, sigma):
        # c_out = -sigma
        # c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1.0 / ((sigma ** 2 + 1.0) ** 0.5)
        return c_in

    def diffusion_predict(self, latent_noise, sigma, positive_tensor, negative_tensor, cond_scale):
        c_out = -sigma
        c_in =  append_dims(self.get_scalings(sigma), latent_noise.ndim)
        t = self.sigma_to_t(sigma)

        # do diffusion_model.forward(x, timesteps=None, context=None, y=None, control=None)
        # with torch.no_grad():
        #     eps1 = self.diffusion_model(latent_noise * c_in, timesteps=t,
        #                 context = positive_tensor['text_encoded'], y=positive_tensor['adm_encoded'], control=None)
        #     eps2 = self.diffusion_model(latent_noise * c_in, timesteps=t,
        #                 context = negative_tensor['text_encoded'], y=negative_tensor['adm_encoded'], control=None)

        x2 = torch.cat((latent_noise * c_in, latent_noise * c_in), dim=0)
        t2 = torch.cat((t, t), dim=0)
        c2 = torch.cat((positive_tensor['text_encoded'], negative_tensor['text_encoded']), dim=0)
        y2 = torch.cat((positive_tensor['adm_encoded'], negative_tensor['adm_encoded']), dim=0)
        with torch.no_grad():
            e2 = self.diffusion_model(x2, timesteps=t2, context=c2, y = y2, control=None)
            eps1 = e2[0:1, :, :, :]
            eps2 = e2[1:2, :, :, :]

        eps = eps2 + (eps1 - eps2) * cond_scale # uncond + (cond - uncond) * cond_scale, get_eps
        # print(f"\n t = {t}")
        # todos.debug.output_var("eps1", eps1)
        # todos.debug.output_var("eps2", eps2)
        # todos.debug.output_var("eps", eps)

        return latent_noise + eps * c_out

    def set_steps(self, steps, denoise=1.0):
        if denoise > 0.9999:
            sigmas = get_karras_sigmas(steps)
        else:
            denoise = max(0.01, denoise)
            new_steps = int(steps/denoise)
            sigmas = get_karras_sigmas(new_steps)
            sigmas = sigmas[-(steps + 1):]
        return sigmas

    def forward(self, positive_tensor, negative_tensor, latent_image, cond_scale=7.5, steps=20, denoise=1.0, seed=-1):
        B, C, H, W = latent_image.size()
        positive_tensor["adm_encoded"] = self.encode_adm(positive_tensor['pooled_output'], B, H, W, positive=True)
        negative_tensor["adm_encoded"] = self.encode_adm(negative_tensor['pooled_output'], B, H, W, positive=False)

        todos.debug.output_var("positive_tensor", positive_tensor)
        todos.debug.output_var("negative_tensor", negative_tensor)

        sigmas = self.set_steps(steps, denoise).to(latent_image.device) # steps, denois ==> sigmas
        # todos.debug.output_var("sigmas", sigmas)
        # print("sigmas: ", sigmas)

        latent_image = self.process_latent_in(latent_image)
        latent_noise = latent_image + prepare_noise(latent_image, seed) * sigmas[0]
        todos.debug.output_var("latent_noise", latent_noise)

        # forget:  steps=20, denoise=1.0, seed=-1
        # forward: latent_noise, positive_tensor, negative_tensor, cond_scale

        # https://github.com/lllyasviel/Fooocus
        # DPM family seems well-suited for XL, since XL sometimes generates overly smooth texture but DPM family sometimes 
        # generate overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.

        # sample_dpm2_ancestral

        # sample = self.sample_euler_ancestral(sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale)
        sample = self.sample_euler(sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale)

        latent_output = self.process_latent_out(sample) # sample

        return latent_output


    def process_latent_in(self, latent):
        return latent * self.scale_factor

    def process_latent_out(self, latent):
        return latent / self.scale_factor

    def sample_euler_ancestral(self, sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale): 
        """Ancestral sampling with Euler method steps."""

        print("sample_euler_ancestral cond_scale: ", cond_scale)
        todos.debug.output_var("sample_euler_ancestral positive_tensor", positive_tensor)
        todos.debug.output_var("sample_euler_ancestral negative_tensor", negative_tensor)

        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            denoised = self.diffusion_predict(latent_noise, sigmas[i] * s_in, positive_tensor, negative_tensor, cond_scale)

            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            d = to_d(latent_noise, sigmas[i], denoised) 

            # Euler method
            dt = sigma_down - sigmas[i]
            latent_noise = latent_noise + d * dt
            if sigmas[i + 1] > 0:
                latent_noise = latent_noise + torch.randn_like(latent_noise) * sigma_up

        todos.debug.output_var("sample_euler_ancestral latent_noise", latent_noise)

        # Bad ???
        # sample_euler_ancestral positive_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 1280], min: -66.179535, max: 18.368391, mean: 0.034937
        #     tensor [pooled_output] size: [1, 1280], min: -4.360085, max: 4.350136, mean: 0.004742
        #     tensor [adm_encoded] size: [1, 2560], min: -4.360085, max: 4.350136, mean: 0.212669
        # sample_euler_ancestral negative_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 1280], min: -66.179535, max: 18.368391, mean: 0.029526
        #     tensor [pooled_output] size: [1, 1280], min: -3.587068, max: 3.409502, mean: 0.024002
        #     tensor [adm_encoded] size: [1, 2560], min: -3.587068, max: 3.409502, mean: 0.230698
        # tensor [sample_euler_ancestral latent_noise] size: [1, 4, 75, 57], min: -2.753332, max: 3.654438, mean: 0.038793


        # OK
        # tensor [sample_euler_ancestral extra_args['cond'][0][0]] size: [1, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.034937
        # sample_euler_ancestral extra_args['cond'][0][1] is dict:
        #     tensor [pooled_output] size: [1, 1280], min: -4.360083, max: 4.350136, mean: 0.004742
        #     tensor [adm_encoded] size: [1, 2560], min: -4.360083, max: 4.350136, mean: 0.185415
        # tensor [sample_euler_ancestral extra_args['uncond'][0][0]] size: [1, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.029526
        # sample_euler_ancestral extra_args['uncond'][0][1] is dict:
        #     tensor [pooled_output] size: [1, 1280], min: -3.58707, max: 3.409507, mean: 0.024002
        #     tensor [adm_encoded] size: [1, 2560], min: -3.58707, max: 3.409507, mean: 0.203443
        # tensor [sample_euler_ancestral latent_noise] size: [1, 4, 75, 57], min: -3.047589, max: 3.609657, mean: -0.01201

        return latent_noise.to(torch.float32)

    def sample_euler(self, sigmas, latent_noise, positive_tensor, negative_tensor, cond_scale):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""

        print("sample_euler cond_scale: ", cond_scale)
        todos.debug.output_var("sample_euler positive_tensor", positive_tensor)
        todos.debug.output_var("sample_euler negative_tensor", negative_tensor)

        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            sigma_hat = sigmas[i]

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            denoised = self.diffusion_predict(latent_noise, sigmas[i] * s_in, positive_tensor, negative_tensor, cond_scale)

            d = to_d(latent_noise, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            latent_noise = latent_noise + d * dt

        todos.debug.output_var("sample_euler latent_noise", latent_noise)

        # Bad
        # sample_euler positive_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 1280], min: -66.179535, max: 18.368391, mean: 0.034937
        #     tensor [pooled_output] size: [1, 1280], min: -4.360085, max: 4.350136, mean: 0.004742
        #     tensor [adm_encoded] size: [1, 2560], min: -4.360085, max: 4.350136, mean: 0.185756
        # sample_euler negative_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 1280], min: -66.179535, max: 18.368391, mean: 0.029526
        #     tensor [pooled_output] size: [1, 1280], min: -3.587068, max: 3.409502, mean: 0.024002
        #     tensor [adm_encoded] size: [1, 2560], min: -3.587068, max: 3.409502, mean: 0.203443
        # tensor [sample_euler latent_noise] size: [1, 4, 75, 57], min: -2.255018, max: 3.478346, mean: 0.150027


        # OK
        # tensor [sample_euler extra_args['cond'][0][0]] size: [1, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.034937
        # sample_euler extra_args['cond'][0][1] is dict:
        #     tensor [pooled_output] size: [1, 1280], min: -4.360083, max: 4.350136, mean: 0.004742
        #     tensor [adm_encoded] size: [1, 2560], min: -4.360083, max: 4.350136, mean: 0.185415
        # tensor [sample_euler extra_args['uncond'][0][0]] size: [1, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.029526
        # sample_euler extra_args['uncond'][0][1] is dict:
        #     tensor [pooled_output] size: [1, 1280], min: -3.58707, max: 3.409507, mean: 0.024002
        #     tensor [adm_encoded] size: [1, 2560], min: -3.58707, max: 3.409507, mean: 0.203443
        # tensor [sample_euler latent_noise] size: [1, 4, 75, 57], min: -3.116404, max: 3.538792, mean: -0.01038

        return latent_noise.to(torch.float32)


    def encode_adm(self, pooled, B, H, W, positive=True):
        return pooled
        # crop_h = 0
        # crop_w = 0
        # aesthetic_score = 6.0 if positive else 2.5

        # out = []
        # out.append(self.embedder(torch.Tensor([H * 8]))) # H * 8 -- 600
        # out.append(self.embedder(torch.Tensor([W * 8]))) # W * 8 -- 456
        # out.append(self.embedder(torch.Tensor([crop_h])))
        # out.append(self.embedder(torch.Tensor([crop_w])))
        # out.append(self.embedder(torch.Tensor([aesthetic_score])))
        # flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(pooled.shape[0], 1)

        # return torch.cat((pooled, flat.to(pooled.device)), dim=1)

def create_ksampler_model(version):
    model = KSampler(version=version)
    model = model.eval()
    # model = torch.jit.script(model)
    model = model.cuda()
    return model

def test():
    # torch.backends.cudnn.enabled = True

    model = create_ksampler_model(version="refiner_1.0")
    # NO load weights, just test process ... !!!!!!!!!!!!!!!!!!!!!!!!

    positive_tensor = {
        "text_encoded" : torch.randn(1, 77, 1280).cuda(),
        "pooled_output" : torch.randn(1, 1280).cuda(),
    }

    negative_tensor = {
        "text_encoded" : torch.randn(1, 77, 1280).cuda(),
        "pooled_output" : torch.randn(1, 1280).cuda(),
    }

    latent_image = torch.randn(1, 4, 75, 57).cuda()
    cond_scale = 7.5
    steps = 200
    denoise = 0.2
    seed = -1

    with torch.no_grad():
        sample_out = model(positive_tensor, negative_tensor, latent_image, cond_scale, steps, denoise, seed)

    todos.debug.output_var("sample_out", sample_out)

 

if __name__ == "__main__":
    test()
