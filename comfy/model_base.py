import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbedNoiseAugmentation
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
from comfy.ldm.modules.diffusionmodules.openaimodel import Timestep
import comfy.model_management
import numpy as np
from enum import Enum
from . import utils
import todos
import pdb

class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2

class BaseModel(torch.nn.Module):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.register_schedule(given_betas=None, beta_schedule=model_config.beta_schedule, timesteps=1000, linear_start=0.00085, linear_end=0.012, cosine_s=8e-3)
        if not unet_config.get("disable_unet_model_creation", False):
            self.diffusion_model = UNetModel(**unet_config, device=device)
        self.model_type = model_type
        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0
        print("model_type", model_type.name)
        print("adm", self.adm_channels)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}):
        # x -- noise_latent_mixer
        todos.debug.output_var("x", x)
        todos.debug.output_var("t", t)
        todos.debug.output_var("c_concat", c_concat)
        todos.debug.output_var("c_crossattn", c_crossattn)
        todos.debug.output_var("c_adm", c_adm)
        todos.debug.output_var("control", control)
        todos.debug.output_var("transformer_options", transformer_options)

        # for refine model
        # tensor [x] size: [2, 4, 75, 57], min: -3.019421, max: 3.357514, mean: -0.024709 ????
        # 
        # tensor [t] size: [2], min: 797.0, max: 797.0
        # [c_concat] value: None
        # tensor [c_crossattn] size: [2, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.032304, positive_tensor ???
        # tensor [c_adm] size: [2, 2560], min: -3.958434, max: 3.409507, mean: 0.195633
        # [control] value: None
        # transformer_options is dict:
        #     list [cond_or_uncond] len: 2
        #     [item] value: '1'
        #     [item] value: '0'
        # self.latent_format.scale_factor -- 0.13025

        # for ClipVision
        # tensor [x] size: [4, 4, 128, 128], min: -2.754989, max: 2.891286, mean: -0.027223
        # tensor [t] size: [4], min: 0.0, max: 0.0
        # tensor [c_crossattn] size: [4, 77, 2048], min: 0.0, max: 0.0, mean: 0.0
        # tensor [c_adm] size: [4, 2816], min: -0.999965, max: 1.0, mean: 0.154741
        # [control] value: None

        if c_concat is not None:
            pdb.set_trace()
            xc = torch.cat([x] + [c_concat], dim=1)
        else:
            xc = x
        context = c_crossattn
        dtype = self.get_dtype()
        xc = xc.to(dtype)
        t = t.to(dtype)
        context = context.to(dtype)
        if c_adm is not None:
            c_adm = c_adm.to(dtype)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # model_forward
        #    self.diffusion_model -- UNetModel.forward(...)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        return self.diffusion_model(xc, t, context=context, y=c_adm, control=control, transformer_options=transformer_options).float()

    def get_dtype(self):
        return self.diffusion_model.dtype

    def is_adm(self):
        return self.adm_channels > 0

    # def encode_adm(self, **kwargs):
    #     return None

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)

        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            print("unet missing:", m)

        if len(u) > 0:
            print("unet unexpected:", u)
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(self, clip_state_dict, vae_state_dict):
        clip_state_dict = self.model_config.process_clip_state_dict_for_saving(clip_state_dict)
        unet_sd = self.diffusion_model.state_dict()
        unet_state_dict = {}
        for k in unet_sd:
            unet_state_dict[k] = comfy.model_management.resolve_lowvram_weight(unet_sd[k], self.diffusion_model, k)

        unet_state_dict = self.model_config.process_unet_state_dict_for_saving(unet_state_dict)
        vae_state_dict = self.model_config.process_vae_state_dict_for_saving(vae_state_dict)
        if self.get_dtype() == torch.float16:
            clip_state_dict = utils.convert_sd_to(clip_state_dict, torch.float16)
            vae_state_dict = utils.convert_sd_to(vae_state_dict, torch.float16)

        if self.model_type == ModelType.V_PREDICTION:
            unet_state_dict["v_pred"] = torch.tensor([])

        return {**unet_state_dict, **vae_state_dict, **clip_state_dict}

    def set_inpaint(self):
        self.concat_keys = ("mask", "masked_image")

def unclip_adm(unclip_conditioning, device, noise_augmentor, noise_augment_merge=0.0):
    # adm_inputs = []
    # noise_aug = []
    for unclip_cond in unclip_conditioning:
        for adm_cond in unclip_cond["clip_vision_output"].image_embeds:
            weight = unclip_cond["strength"]
            noise_augment = unclip_cond["noise_augmentation"]
            noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
            c_adm, noise_level_emb = noise_augmentor(adm_cond.to(device), 
                noise_level=torch.tensor([noise_level], device=device))
            adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
            # noise_aug.append(noise_augment)
            # adm_inputs.append(adm_out)

    # if len(noise_aug) > 1:
    #     pdb.set_trace()
    #     adm_out = torch.stack(adm_inputs).sum(0)
    #     noise_augment = noise_augment_merge
    #     noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
    #     c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim], 
    #         noise_level=torch.tensor([noise_level], device=device))
    #     adm_out = torch.cat((c_adm, noise_level_emb), 1)

    return adm_out

class SD21UNCLIP(BaseModel):
    def __init__(self, model_config, noise_aug_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = CLIPEmbedNoiseAugmentation(**noise_aug_config)

    def encode_adm(self, **kwargs):
        unclip_conditioning = kwargs.get("unclip_conditioning", None)
        device = kwargs["device"]
        if unclip_conditioning is None:
            return torch.zeros((1, self.adm_channels))
        else:
            return unclip_adm(unclip_conditioning, device, self.noise_augmentor, kwargs.get("unclip_noise_augment_merge", 0.05))

def sdxl_pooled(args, noise_augmentor):
    if "unclip_conditioning" in args:
        # revision ==> pdb.set_trace()
        return unclip_adm(args.get("unclip_conditioning", None), args["device"], noise_augmentor)[:,:1280]
    else:
        return args["pooled_output"]

class SDXLRefiner(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        # self.noise_augmentor = CLIPEmbedNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        # for refine model
        # kwargs.keys() -- ['device', 'pooled_output', 'width', 'height', 'prompt_type']
        # kwargs -- {'device': device(type='cuda', index=0), 'pooled_output': tensor([[-0.087919, -1.548311, -0.571016,  ...,  0.208579, -2.006104,
        #  -0.290721]]), 'width': 456, 'height': 600, 'prompt_type': 'positive'}

        clip_pooled = kwargs["pooled_output"] # xxxx8888 sdxl_pooled(kwargs, self.noise_augmentor)
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

class SDXL(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbedNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

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
