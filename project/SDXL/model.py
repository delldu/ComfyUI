# import torch
# from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
# from comfy.ldm.modules.encoders.noise_aug_modules import EmbedNoiseAugmentation
# from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
# from comfy.ldm.modules.diffusionmodules.openaimodel import Timestep
# import comfy.model_management
# from enum import Enum
# from . import utils


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

import todos
import pdb

class BaseModel(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.scale_factor = 0.13025
        self.diffusion_model = UNetModel(version=version)
        self.register_schedule(beta_schedule="linear", timesteps=1000, 
            linear_start=0.00085, linear_end=0.012)

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

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))

    def forward(self, x, t, c_crossattn=None, c_adm=None, control=None):
        print("CheckPoint BaseModel.forward -- diffusion_model, get_dtype()=", self.diffusion_model.dtype)

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


def unclip_adm(unclip_condition, device, noise_augmentor, noise_augment_merge=0.0):
    # unclip_condition[0].keys() -- ['clip_vision_output', 'strength', 'noise_augmentation']

    # unclip_condition[0]['clip_vision_output']
    # unclip_condition[0]['clip_vision_output'].image_embeds.size() -- [1, 1280]
    # unclip_condition[0]['clip_vision_output'].last_hidden_state.size() -- [1, 257, 1664]
    # unclip_condition[0]['clip_vision_output'].hidden_states -- None
    # unclip_condition[0]['clip_vision_output'].attentions -- None

    # unclip_condition[0]['strength'] -- 1.0
    # unclip_condition[0]['noise_augmentation'] -- 0.0

    # noise_augmentor
    # EmbedNoiseAugmentation(
    #   (time_embed): Timestep()
    # )
    # noise_augmentor.max_noise_level -- 1000.0


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
    def __init__(self, version="base_1.0"):
        super().__init__(version=version)
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
    def __init__(self, version="refiner_1.0"):
        super().__init__(version=version)
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


if __name__ == "__main__":
    model = SDXL()
    model = model.eval()
    # model = torch.jit.script(model)

    x = torch.randn(1, 4, 146, 111)
    t = torch.randn(1)
    c_crossattn = torch.randn(1, 77, 2048)
    c_adm = torch.randn(1, 2816)
    control = {
        "input":[],
        "middle": [torch.randn(1, 1280, 37, 28)],
        "output": [torch.randn(1, 320, 146, 111), torch.randn(1, 320, 146, 111), torch.randn(1, 320, 146, 111),
            torch.randn(1, 320, 73, 56), torch.randn(1, 640, 73, 56), torch.randn(1, 640, 73, 56),
            torch.randn(1, 640, 37, 28), torch.randn(1, 1280, 37, 28), torch.randn(1, 1280, 37, 28),
        ],
    }

    # pdb.set_trace()
    # model.diffusion_model.dtype

    with torch.no_grad():
        output = model(x, t, c_crossattn=c_crossattn, c_adm=c_adm, control=control)

    print(output.size()) # [1, 4, 146, 111]
    # print(model)
