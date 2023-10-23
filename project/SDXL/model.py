import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange

from SDXL.unet import (
    Timestep,
    UNetModel,
)
from SDXL.util import (
    make_beta_schedule,
)
from SDXL.noise import (
    CLIPEmbedNoiseAugmentation,
)
from SDXL.vae import (
    VAEEncode,
    VAEDecode,
)
from SDXL.clip import (
    CLIPTextEncode,
)
from SDXL.tokenizer import (
    CLIPTextTokenizer,
)

import todos
import pdb


def prepare_noise(latent_image, seed):
    generator = torch.manual_seed(seed)
    return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, 
        device=latent_image.device)


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

class SDXLRefiner(nn.Module):
    def __init__(self, version="refiner_1.0"):
        super(SDXLRefiner, self).__init__()
        self.scale_factor = 0.13025
        self.diffusion_model = nn.Identity() # UNetModel(version=version)
        self.vae_encode_model = nn.Identity() # VAEEncode()
        self.vae_decode_model = nn.Identity() # VAEDecode()
        self.clip_text_model = CLIPTextEncode(version=version)

        # Option models ...
        self.clip_vision_model = nn.Identity()
        self.control_lora_model = nn.Identity()

        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbedNoiseAugmentation()

        self.register_schedule(beta_schedule="linear", timesteps=1000, linear_start=0.00085, linear_end=0.012)

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

    def get_eps(self, latent_noise, sigma, **kwargs):
        # tensor [input] size: [1, 4, 75, 57], min: -3.141021, max: 3.365364, mean: -0.027016
        # tensor [sigma] size: [1], min: 0.149319, max: 0.149319, mean: 0.149319
        # kwargs.keys() -- ['cond', 'uncond', 'cond_scale', 'cond_concat', 'model_options', 'seed']

        # sigma -- tensor([0.149319], device='cuda:0')
        # ==> 
        # self.get_scalings(sigma) -- (tensor([-0.149319], device='cuda:0'), tensor([0.989035], device='cuda:0'))
        # ==> self.sigma_to_t(sigma) -- 23

        # c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_out = -sigma
        c_in =  append_dims(self.get_scalings(sigma), latent_noise.ndim)
        eps = latent_noise # self.get_eps(latent_noise * c_in, self.sigma_to_t(sigma), **kwargs)

        return latent_noise + eps * c_out


    def set_steps(self, steps, denoise=1.0):
        denoise = min(0.1, denoise)
        if denoise > 0.9999:
            sigmas = get_karras_sigmas(steps)
        else:
            denoise = min(denoise, 0.01)
            new_steps = int(steps/denoise)
            sigmas = get_karras_sigmas(new_steps)
            sigmas = sigmas[-(steps + 1):]
        return sigmas


    def start_sample(self, positive, negative, latent, steps, denoise, cfg=7.5, seed=-1):
        latent = model.process_latent_in(latent)
        latent_noise = prepare_noise(latent, seed) # latent, seed ==> latent_noise

        sigmas = self.set_steps(steps, denoise).to(latent.device) # steps, denois ==> sigmas
        sample = self.sample_euler_ancestral(latent, sigmas)

        return latent # self.process_latent_out(sample)


    def forward_x(self, x, t, c_crossattn=None, c_adm=None, control=None):
        x = x.to(self.diffusion_model.dtype)
        t = t.to(self.diffusion_model.dtype)
        context = c_crossattn.to(self.diffusion_model.dtype)
        c_adm = c_adm.to(self.diffusion_model.dtype)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # model_forward
        #    self.diffusion_model -- UNetModel.forward(...)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        return self.diffusion_model(x, t, context=context, y=c_adm, control=control).float()

    def process_latent_in(self, latent):
        return latent * self.scale_factor

    def process_latent_out(self, latent):
        return latent / self.scale_factor


    def sample_euler_ancestral(self, latent_noise, sigmas, extra_args=None):
        """Ancestral sampling with Euler method steps."""

        # xxxx_refiner 1
        extra_args = {}
        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            # model --
            #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            with torch.no_grad():
                denoised = self.diffusion_model(latent_noise) # , sigmas[i] * s_in, **extra_args)
            # denoised = self.get_eps(latent_noise, sigmas[i], ...)

            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            d = to_d(latent_noise, sigmas[i], denoised) 

            # Euler method
            dt = sigma_down - sigmas[i]
            latent_noise = latent_noise + d * dt
            if sigmas[i + 1] > 0:
                latent_noise = latent_noise + torch.randn_like(latent_noise) * sigma_up
        return latent_noise

    def sample_euler(model, latent_noise, sigmas, extra_args=None, s_churn=0.):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""

        extra_args = {} if extra_args is None else extra_args
        s_in = latent_noise.new_ones([latent_noise.shape[0]])
        for i in trange(len(sigmas) - 1):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(latent_noise)
                latent_noise = latent_noise + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            # model --
            #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            with torch.no_grad():
                denoised = self.diffusion_model(latent_noise) # , sigmas[i] * s_in, **extra_args)

            d = to_d(latent_noise, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            latent_noise = latent_noise + d * dt
        return latent_noise


    def encode_adm(self, **kwargs):
        # kwargs.keys() -- ['device', 'pooled_output', 'unclip_condition', 'width', 'height', 'prompt_type']
        
        clip_pooled = kwargs["pooled_output"]
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)

        if kwargs.get("prompt_type", "") == "negative":
            aesthetic_score = kwargs.get("aesthetic_score", 2.5)
        else:
            aesthetic_score = kwargs.get("aesthetic_score", 6)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


def test_refiner():
    model = SDXLRefiner()
    model = model.eval()
    # model = torch.jit.script(model)
    model.cuda()

    positive_prompt = "bag, clean background, made from cloth"
    negative_prompt = "watermark, text"

    clip_token = CLIPTextTokenizer(version="refiner_1.0")
    positive_tokens = clip_token.encode(positive_prompt)

    print(positive_tokens)

    {'g': [49406, 3365, 267, 3772, 5994, 267, 1105, 633, 14559, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}


    # (Pdb) tokens['l']
    # [[(49406, 1.0), (3365, 1.0), (267, 1.0), (3772, 1.0), (5994, 1.0), (267, 1.0), (1105, 1.0), (633, 1.0), (14559, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0)]]
    # (Pdb) tokens['g']
    # [[(49406, 1.0), (3365, 1.0), (267, 1.0), (3772, 1.0), (5994, 1.0), (267, 1.0), (1105, 1.0), (633, 1.0), (14559, 1.0), 
    # (49407, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0)]]


    negative_tokens = clip_token.encode(negative_prompt)
    print(negative_tokens)

    # (Pdb) tokens['l']
    # [[(49406, 1.0), (2505, 1.0), (2110, 1.0), (267, 1.0), (4160, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0), (49407, 1.0)]]
    # (Pdb) tokens['g']
    # [[(49406, 1.0), (2505, 1.0), (2110, 1.0), (267, 1.0), (4160, 1.0), (49407, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0)]]

    positive_tensor, positive_pooled = model.clip_text_model(torch.LongTensor(positive_tokens['g']).cuda())
    todos.debug.output_var("positive_tensor", positive_tensor)
    todos.debug.output_var("positive_pooled", positive_pooled)
    # tensor [positive_tensor] size: [1, 77, 1280], min: -28.947594, max: 31.752134, mean: 0.09212
    # tensor [positive_pooled] size: [1, 1280], min: -3.62947, max: 3.656186, mean: 0.001459

    negative_tensor, negative_pooled = model.clip_text_model(torch.LongTensor(negative_tokens['g']).cuda())
    todos.debug.output_var("negative_tensor", negative_tensor)
    todos.debug.output_var("negative_pooled", negative_pooled)
    # tensor [negative_tensor] size: [1, 77, 1280], min: -28.947594, max: 29.09692, mean: 0.090274
    # tensor [negative_pooled] size: [1, 1280], min: -3.919105, max: 3.43405, mean: 0.020215

    pdb.set_trace()




    # x = torch.randn(1, 4, 146, 111).cuda()
    # t = torch.randn(1).cuda()
    # c_crossattn = torch.randn(1, 77, 1280).cuda()
    # c_adm = torch.randn(1, 2560).cuda()
    # control = None

    # sigmas = model.set_steps(10, denoise=0.2).to(x.device)
    # output = model.sample_euler_ancestral(x, sigmas)
    # # with torch.no_grad():
    # #     output = model(x, t, c_crossattn=c_crossattn, c_adm=c_adm, control=control)

    # print(output.size()) # [1, 4, 146, 111]
    # print(model)


if __name__ == "__main__":
    test_refiner()
