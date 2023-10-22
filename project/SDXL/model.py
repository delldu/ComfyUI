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
    EmbedNoiseAugmentation,
)
from SDXL.vae import (
    VAEEncode,
    VAEDecode,
)
from SDXL.clip import (
    SDXLCLIPTextModel,
)

import todos
import pdb

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

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

# KarrasScheduler -- sigma_min, sigma_max ...
def get_karras_sigmas(n, sigma_min=0.0291675, sigma_max=14.614642, rho=7.0):
    """Constructs the noise schedule of Karras et al. (2022)."""

    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho # size() -- 13

    return torch.cat([sigmas, sigmas.new_zeros([1])])

class BaseModel(nn.Module):
    def __init__(self, version):
        super(BaseModel, self).__init__()
        self.scale_factor = 0.13025
        self.diffusion_model = nn.Identity() # UNetModel(version=version)
        self.vae_encode_model = nn.Identity() # VAEEncode()
        self.vae_decode_model = nn.Identity() # VAEDecode()
        self.clip_text_model = nn.Identity() # SDXLCLIPTextModel(version=version)

        # Option models ...
        self.clip_vision_model = nn.Identity()
        self.control_lora_model = nn.Identity()

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

    def forward(self, x, t, c_crossattn=None, c_adm=None, control=None):
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


    def euler_ancestral_sample(self, x, sigmas, extra_args=None):
        """Ancestral sampling with Euler method steps."""

        # xxxx_refiner 1
        extra_args = {}
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # model_forward
            # model --
            #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            with torch.no_grad():
                denoised = self.diffusion_model(x) # , sigmas[i] * s_in, **extra_args)

            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
            d = to_d(x, sigmas[i], denoised) 

            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up
        return x




def unclip_adm(unclip_condition, device, noise_augmentor, noise_augment_merge=0.0):
    adm_inputs = []
    noise_aug = []
    for unclip_cond in unclip_condition:
        for adm_cond in unclip_cond["clip_vision_output"].image_embeds:
            weight = unclip_cond["strength"] # 1.0
            noise_augment = unclip_cond["noise_augmentation"]
            noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment) # -- 0
            c_adm, noise_level_emb = noise_augmentor(adm_cond.to(device), 
                noise_level=torch.tensor([noise_level], device=device))
            # noise_level_emb.size() -- [1, 1280]

            adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
            noise_aug.append(noise_augment)
            adm_inputs.append(adm_out)

    if len(noise_aug) > 1: # False
        pdb.set_trace()
        adm_out = torch.stack(adm_inputs).sum(0)
        noise_augment = noise_augment_merge
        noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
        c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim], noise_level=torch.tensor([noise_level], device=device))
        adm_out = torch.cat((c_adm, noise_level_emb), 1)

    # adm_out.size() -- [1, 2560]

    return adm_out


def sdxl_pooled(args, noise_augmentor):
    # noise_augmentor -- EmbedNoiseAugmentation((time_embed): Timestep())

    if "unclip_condition" in args: # False
        # ClipVision
        return unclip_adm(args.get("unclip_condition", None), args["device"], noise_augmentor)[:,:1280]
    else:
        return args["pooled_output"]


class SDXL(BaseModel):
    def __init__(self):
        super(SDXL, self).__init__(version="base_1.0")
        self.embedder = Timestep(256)
        self.noise_augmentor = EmbedNoiseAugmentation()

    def encode_adm(self, **kwargs):
        # kwargs -- 
        # {'device': device(type='cuda', index=0), 'pooled_output': tensor([[ 0.5246, -0.2884, -0.2883,  ..., -0.6900,  1.4370, -1.0179]]), 'control': <comfy.controlnet.ControlLora object at 0x7fee48d770a0>, 
        #     'width': 1256, 'height': 832, 'prompt_type': 'positive'}

        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class SDXLRefiner(BaseModel):
    def __init__(self):
        super(SDXLRefiner, self).__init__(version="refiner_1.0")
        self.embedder = Timestep(256)
        self.noise_augmentor = EmbedNoiseAugmentation()

    def encode_adm(self, **kwargs):
        # kwargs.keys() -- ['device', 'pooled_output', 'unclip_condition', 'width', 'height', 'prompt_type']
        
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
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

def test_sdxl():
    model = SDXL()
    model = model.eval()
    # model = torch.jit.script(model)
    model.cuda()

    x = torch.randn(1, 4, 146, 111).cuda()
    t = torch.randn(1).cuda()
    c_crossattn = torch.randn(1, 77, 2048).cuda()
    c_adm = torch.randn(1, 2816).cuda()


    control = {
        "input":[],
        "middle": [torch.randn(1, 1280, 37, 28).cuda()],
        "output": [
            torch.randn(1, 320, 146, 111).cuda(), torch.randn(1, 320, 146, 111).cuda(), torch.randn(1, 320, 146, 111).cuda(),
            torch.randn(1, 320, 73, 56).cuda(), torch.randn(1, 640, 73, 56).cuda(), torch.randn(1, 640, 73, 56).cuda(),
            torch.randn(1, 640, 37, 28).cuda(), torch.randn(1, 1280, 37, 28).cuda(), torch.randn(1, 1280, 37, 28).cuda(),
        ],
    }

    with torch.no_grad():
        output = model(x, t, c_crossattn=c_crossattn, c_adm=c_adm, control=control)

    print(output.size()) # [1, 4, 146, 111]
    # print(model)



def test_refiner():
    model = SDXLRefiner()
    model = model.eval()
    # model = torch.jit.script(model)
    model.cuda()

    x = torch.randn(1, 4, 146, 111).cuda()
    t = torch.randn(1).cuda()
    c_crossattn = torch.randn(1, 77, 1280).cuda()
    c_adm = torch.randn(1, 2560).cuda()
    control = None

    sigmas = model.set_steps(10, denoise=0.2).to(x.device)
    output = model.euler_ancestral_sample(x, sigmas)
    # with torch.no_grad():
    #     output = model(x, t, c_crossattn=c_crossattn, c_adm=c_adm, control=control)

    print(output.size()) # [1, 4, 146, 111]
    print(model)


if __name__ == "__main__":
    # test_sdxl()

    test_refiner()
