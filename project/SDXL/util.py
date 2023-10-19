import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class DictToClass(object):
    def __init__(self, _obj):
        if _obj:
            self.__dict__.update(_obj)

def state_dict_load(checkpoint):
    _, extension = os.path.splitext(checkpoint)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(checkpoint, device='cpu')
    else:
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
    return state_dict


def state_dict_filter(source_sd, prefix_list, remove_prefix=False):
    target_sd = {}
    keys = list(source_sd.keys())

    for prefix in prefix_list:
        for k in keys:
            if k.startswith(prefix):
                if remove_prefix:
                    target_sd[k[len(prefix):]] = source_sd.pop(k)
                else:
                    target_sd[k] = source_sd.pop(k)

    return target_sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict

def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
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
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd


def process_base_clip_state_dict(state_dict):
    keys_to_replace = {}
    replace_prefix = {}

    replace_prefix["conditioner.embedders.0.transformer.text_model"] = "cond_stage_model.clip_l.transformer.text_model"
    state_dict = transformers_convert(state_dict, "conditioner.embedders.1.model.", "cond_stage_model.clip_g.transformer.text_model.", 32)
    keys_to_replace["conditioner.embedders.1.model.text_projection"] = "cond_stage_model.clip_g.text_projection"
    keys_to_replace["conditioner.embedders.1.model.logit_scale"] = "cond_stage_model.clip_g.logit_scale"

    state_dict = state_dict_prefix_replace(state_dict, replace_prefix)
    state_dict = state_dict_key_replace(state_dict, keys_to_replace)
    return state_dict


def process_refiner_clip_state_dict(state_dict):
    keys_to_replace = {}
    replace_prefix = {}

    state_dict = transformers_convert(state_dict, "conditioner.embedders.0.model.", "cond_stage_model.clip_g.transformer.text_model.", 32)
    keys_to_replace["conditioner.embedders.0.model.text_projection"] = "cond_stage_model.clip_g.text_projection"
    keys_to_replace["conditioner.embedders.0.model.logit_scale"] = "cond_stage_model.clip_g.logit_scale"

    state_dict = state_dict_key_replace(state_dict, keys_to_replace)
    return state_dict


def load_base_clip_model_weight(model, model_path="models/sd_xl_base_1.0.safetensors"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")

        state_dict = state_dict_load(checkpoint)
        state_dict = process_base_clip_state_dict(state_dict)
        target_state_dict = state_dict_filter(state_dict, ["cond_stage_model."], remove_prefix=True)
        m, u = model.load_state_dict(target_state_dict, strict=False)
        if len(m) > 0:
            print(f"Load weight from {checkpoint} missing keys: ", m)
        if len(u) > 0:
            print(f"Load weight from {checkpoint} leftover keys: ", u)        
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")


def load_refiner_clip_model_weight(model, model_path="models/sd_xl_refiner_1.0.safetensors"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")

        state_dict = state_dict_load(checkpoint)
        state_dict = process_refiner_clip_state_dict(state_dict)
        target_state_dict = state_dict_filter(state_dict, ["cond_stage_model."], remove_prefix=True)
        m, u = model.load_state_dict(target_state_dict, strict=False)
        if len(m) > 0:
            print(f"Load weight from {checkpoint} missing keys: ", m)
        if len(u) > 0:
            print(f"Load weight from {checkpoint} leftover keys: ", u)
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")


def load_model_weight(model, model_path="models/sdxl_vae.safetensors"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")

        state_dict = state_dict_load(checkpoint)
        model.load_state_dict(state_dict)
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")


def load_vae_model_weight(model, model_path="models/sdxl_vae.safetensors"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")

        state_dict = state_dict_load(checkpoint)
        state_dict = state_dict_filter(state_dict, ["encoder.", "quant_conv.",
            "decoder.", "post_quant_conv."], remove_prefix=False)

        model.load_state_dict(state_dict)
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")

def load_vaeencode_model_weight(model, model_path="models/sdxl_vae.safetensors"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")

        state_dict = state_dict_load(checkpoint)
        state_dict = state_dict_filter(state_dict, ["encoder.", "quant_conv."], remove_prefix=False)
        model.load_state_dict(state_dict)
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")

def load_vaedecode_model_weight(model, model_path="models/sdxl_vae.safetensors"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")

        state_dict = state_dict_load(checkpoint)
        state_dict = state_dict_load(checkpoint)
        state_dict = state_dict_filter(state_dict, ["decoder.", "post_quant_conv."], remove_prefix=False)
        model.load_state_dict(state_dict)
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")


def load_diffusion_model_weight(model, model_path="models/sd_xl_base_1.0.safetensors"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")

        state_dict = state_dict_load(checkpoint)
        target_state_dict = state_dict_filter(state_dict, ["model.diffusion_model."], remove_prefix=True)
        model.load_state_dict(target_state_dict)
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return Conv2d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")

# -------------------------

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def make_beta_schedule(n_timestep, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only: # True
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def image_crop_8x8(image):
    B, C, H, W = image.size()
    H8 = (H // 8) * 8
    W8 = (W // 8) * 8
    if H8 != H or W8 != W:
        top = (H % 8) // 2
        left = (W % 8) // 2
        image = image[:, :, top : H8 + top, left : W8 + left]
    return image


if __name__ == "__main__":
    image = torch.randn(2, 3, 1011, 777)
    pad = image_crop_8x8(image)

    pdb.set_trace()
