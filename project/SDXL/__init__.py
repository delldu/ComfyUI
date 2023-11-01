"""Uni-Stable-Diffusion Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#
__version__ = "1.0.0"

import os

import torch
import todos
# from .uni_controlnet import UniControlNet

from SDXL.util import (
    DictToClass,
    load_torch_image,
    state_dict_load,
    state_dict_filter,
    base_clip_state_dict,
    refiner_clip_state_dict,
    load_model_weight,
    load_vaeencode_model_weight,
    load_vaedecode_model_weight,
    load_diffusion_model_weight,        
)

# from SDXL.ksampler import (
#     KSampler,
# )
from SDXL.model import (
    SDXLBase,
    SDXLRefiner,
)

from SDXL.vae import (
    VAEEncode,
    VAEDecode,
)

from SDXL.clip import (
    CLIPTextEncode,
    CLIPVisionEncode,
    # clip_vision_model,
)

from SDXL.tokenizer import (
    CLIPTextTokenizer,
)

from SDXL.controlnet import (
    load_ctrl_lora_weights,
)

import pdb

# import json
# d = {
#     "positive": "a women is skiing on top of mountain",
#     "negative": "black and white",
#     "guidance": 8.0,
#     "sampler": "karras/euler_a",
#     "seed": 65535,
#     "steps": 30,
#     "denoise": 1.0,
# }
# s = json.dumps(d)
# d = json.loads(s)



def create_model(version):
    """
    Create model
    """

    model = UniControlNet(version=version)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model {version} on {device} ...")

    return model, device


def get_model(version):
    """Load jit script model."""

    model, device = create_model(version)
    # print(model)

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);
    
    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # torch_file_name = f"output/{version}.torch"
    # if not os.path.exists(torch_file_name):
    #     model.save(torch_file_name)

    return model, device


def create_sdxl_base_model():
    model_version = "base_1.0"
    model_path = "models/sd_xl_base_1.0.safetensors"
    model = DictToClass({
        "sample_mode": SDXLBase(),
        "vae_encode": VAEEncode(version=model_version),
        "vae_decode": VAEDecode(version=model_version),
        "clip_token": CLIPTextTokenizer(version=model_version),
        "clip_text": CLIPTextEncode(version=model_version),
        "clip_vision": CLIPVisionEncode(),
    })
    whole_sd = state_dict_load(model_path)

    model_sd = state_dict_filter(whole_sd, ["model.diffusion_model."], remove_prefix=True)
    model.sample_mode.diffusion_model.load_state_dict(model_sd)
    model.sample_mode.diffusion_model = model.sample_mode.diffusion_model.eval()
    # load_diffusion_model_weight(model.sample_mode.diffusion_model, model_path="models/sd_xl_refiner_1.0.safetensors")
    load_ctrl_lora_weights(model.sample_mode.lora_model, model_path="models/control-lora-canny-rank128.safetensors", 
        unet_weight=model_sd)

    model.sample_mode = model.sample_mode.half().eval().cuda()
    # model.sample_mode = model.sample_mode.eval().cuda()

    vae_sd = state_dict_filter(whole_sd, ["first_stage_model."], remove_prefix=True)
    model_sd = state_dict_filter(vae_sd, ["encoder.", "quant_conv."], remove_prefix=False)
    model.vae_encode.load_state_dict(model_sd)
    # load_vaeencode_model_weight(model.vae_encode, model_path="models/sdxl_vae.safetensors")
    model.vae_encode = model.vae_encode.eval()

    model_sd = state_dict_filter(vae_sd, ["decoder.", "post_quant_conv."], remove_prefix=False)
    model.vae_decode.load_state_dict(model_sd)
    # load_vaedecode_model_weight(model.vae_decode, model_path="models/sdxl_vae.safetensors")
    model.vae_decode = model.vae_decode.eval()

    # model.clip_token load weight by self

    model_sd = base_clip_state_dict(whole_sd)
    m, u = model.clip_text.load_state_dict(model_sd, strict=False)
    if len(m) > 0:
        print(f"CLIPTextEncode load weight missing keys: ", m)
    if len(u) > 0:
        print(f"CLIPTextEncode load weight leftover keys: ", u)
    model.clip_text = model.clip_text.eval()

    model.clip_vision.eval()

    return model


def create_sdxl_refiner_model():
    model_version = "refiner_1.0"
    model_path = "models/sd_xl_refiner_1.0.safetensors"
    model = DictToClass({
        "sample_mode": SDXLRefiner(),
        "vae_encode": VAEEncode(version=model_version),
        "vae_decode": VAEDecode(version=model_version),
        "clip_token": CLIPTextTokenizer(version=model_version),
        "clip_text": CLIPTextEncode(version=model_version),
        "clip_vision": None,
    })
    whole_sd = state_dict_load(model_path)

    model_sd = state_dict_filter(whole_sd, ["model.diffusion_model."], remove_prefix=True)
    model.sample_mode.diffusion_model.load_state_dict(model_sd)
    model.sample_mode.diffusion_model = model.sample_mode.diffusion_model.eval()
    # load_diffusion_model_weight(model.sample_mode.diffusion_model, model_path="models/sd_xl_refiner_1.0.safetensors")    
    model.sample_mode = model.sample_mode.eval().cuda()
    # model.sample_mode = model.sample_mode.eval().cuda()

    vae_sd = state_dict_filter(whole_sd, ["first_stage_model."], remove_prefix=True)
    model_sd = state_dict_filter(vae_sd, ["encoder.", "quant_conv."], remove_prefix=False)
    model.vae_encode.load_state_dict(model_sd)
    # load_vaeencode_model_weight(model.vae_encode, model_path="models/sdxl_vae.safetensors")
    model.vae_encode = model.vae_encode.eval()

    model_sd = state_dict_filter(vae_sd, ["decoder.", "post_quant_conv."], remove_prefix=False)
    model.vae_decode.load_state_dict(model_sd)
    # load_vaedecode_model_weight(model.vae_decode, model_path="models/sdxl_vae.safetensors")
    model.vae_decode = model.vae_decode.eval()

    # model.clip_token load weight by self

    model_sd = refiner_clip_state_dict(whole_sd)
    m, u = model.clip_text.load_state_dict(model_sd, strict=False)
    if len(m) > 0:
        print(f"CLIPTextEncode load weight missing keys: ", m)
    if len(u) > 0:
        print(f"CLIPTextEncode load weight leftover keys: ", u)
    model.clip_text = model.clip_text.eval()   

    return model
