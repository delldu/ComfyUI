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
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import numpy as np

import pdb


class DictToClass(object):
    def __init__(self, _obj):
        if _obj:
            self.__dict__.update(_obj)

def state_dict_load(model_path):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if not os.path.exists(checkpoint):
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}' not exist !!!")
        return None

    # file exist
    print(f"Loading weight from {checkpoint} ...")

    _, extension = os.path.splitext(checkpoint)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(checkpoint, device="cpu")
    else:
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    return state_dict


def state_dict_filter(source_sd, prefix_list, remove_prefix=False):
    target_sd = {}
    keys = list(source_sd.keys())

    for prefix in prefix_list:
        skip_len = len(prefix) if remove_prefix else 0
        for k in keys:
            if k.startswith(prefix):
                target_sd[k[skip_len:]] = source_sd.pop(k)

    return target_sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix):
    out = state_dict
    for rp in replace_prefix:
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from * x : shape_from * (x + 1)]
    return sd


def base_clip_text_state_dict(state_dict):
    keys_to_replace = {}
    replace_prefix = {}

    replace_prefix["conditioner.embedders.0.transformer.text_model"] = "cond_stage_model.clip_l.transformer.text_model"
    state_dict = transformers_convert(
        state_dict, "conditioner.embedders.1.model.", "cond_stage_model.clip_g.transformer.text_model.", 32
    )
    keys_to_replace["conditioner.embedders.1.model.text_projection"] = "cond_stage_model.clip_g.text_projection"
    keys_to_replace["conditioner.embedders.1.model.logit_scale"] = "cond_stage_model.clip_g.logit_scale"

    state_dict = state_dict_prefix_replace(state_dict, replace_prefix)
    state_dict = state_dict_key_replace(state_dict, keys_to_replace)

    state_dict = state_dict_filter(state_dict, ["cond_stage_model."], remove_prefix=True)

    return state_dict


def refiner_clip_text_state_dict(state_dict):
    keys_to_replace = {}
    replace_prefix = {}

    state_dict = transformers_convert(
        state_dict, "conditioner.embedders.0.model.", "cond_stage_model.clip_g.transformer.text_model.", 32
    )
    keys_to_replace["conditioner.embedders.0.model.text_projection"] = "cond_stage_model.clip_g.text_projection"
    keys_to_replace["conditioner.embedders.0.model.logit_scale"] = "cond_stage_model.clip_g.logit_scale"
    state_dict = state_dict_key_replace(state_dict, keys_to_replace)
    state_dict = state_dict_filter(state_dict, ["cond_stage_model."], remove_prefix=True)

    return state_dict


def load_base_clip_text_model_weight(model, model_path="models/sd_xl_base_1.0.safetensors"):
    if os.environ.get("SDXL_LOAD") == "NO":
        return
    state_dict = state_dict_load(model_path)
    target_state_dict = base_clip_text_state_dict(state_dict)
    m, u = model.load_state_dict(target_state_dict, strict=False)
    if len(m) > 0:
        print(f"Load weight from {model_path} missing keys: ", m)
    if len(u) > 0:
        print(f"Load weight from {model_path} leftover keys: ", u)


def load_refiner_clip_text_model_weight(model, model_path="models/sd_xl_refiner_1.0.safetensors"):
    state_dict = state_dict_load(model_path)
    target_state_dict = refiner_clip_text_state_dict(state_dict)
    m, u = model.load_state_dict(target_state_dict, strict=False)
    if len(m) > 0:
        print(f"Load weight from {model_path} missing keys: ", m)
    if len(u) > 0:
        print(f"Load weight from {model_path} leftover keys: ", u)


def load_model_weight(model, model_path="models/sdxl_vae.safetensors"):
    state_dict = state_dict_load(model_path)
    return model.load_state_dict(state_dict)


def load_vae_model_weight(model, model_path="models/sdxl_vae.safetensors"):
    state_dict = state_dict_load(model_path)
    state_dict.pop("model_ema.decay")
    state_dict.pop("model_ema.num_updates")

    model.load_state_dict(state_dict)


def load_unet_model_weight(model, model_path="models/sd_xl_base_1.0.safetensors"):
    state_dict = state_dict_load(model_path)
    target_state_dict = state_dict_filter(state_dict, ["model.diffusion_model."], remove_prefix=True)
    model.load_state_dict(target_state_dict)


# -------------------------


def count_model_params(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print("-" * 120)
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f}M params.")
        print("-" * 120)

    return total_params


def make_beta_schedule(n_timestep: int, linear_start: float = 1e-4, linear_end: float = 2e-2):
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)

    def __repr__(self):
        return f"Timestep({self.dim})"

def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def image_crop_32x32(image):
    B, C, H, W = image.size()
    H32 = H // 32
    W32 = W // 32
    if 32 * H32 != H or 32 * W32 != W:
        image = F.interpolate(image, size=((H32 + 1) * 32, (W32 + 1) * 32), mode="bilinear", align_corners=False)

    return image


def load_image(filename):
    image = Image.open(filename).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,].movedim(-1, 1)  # permute(0, 3, 1, 2) ==> BxCxHxW
    return image_crop_32x32(image)


def load_clip_vision_image(filename):
    image = Image.open(filename).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,].movedim(-1, 1)  # permute(0, 3, 1, 2) ==> BxCxHxW
    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    # image normal
    image = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])(image)

    return image


def load_torch_image(image):
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,].movedim(-1, 1)  # permute(0, 3, 1, 2) ==> BxCxHxW
    # image = F.interpolate(image, size=(1024, 1024), mode="bilinear", align_corners=False)
    return image_crop_32x32(image)


if __name__ == "__main__":
    image = torch.randn(2, 3, 1011, 777)
    pad = image_crop_32x32(image)

    pdb.set_trace()
