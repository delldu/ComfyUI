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
__version__ = "1.0.0"

import torch
import torch.nn as nn

from SDXL.util import (
    DictToClass,
    state_dict_load,
    state_dict_filter,
)

from SDXL.model import (
    SDXLCreator,
    SDXLRefiner,
)

from SDXL.vae import (
    create_vae_model,
)

from SDXL.clip_text import (
    create_clip_text_model,
)
from SDXL.clip_vision import (
    create_clip_vision_model,
)
from SDXL.tokenizer import (
    create_clip_token_model,
)


import todos
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


def create_sdxl_base_model(skip_lora=True, skip_vision=True):
    model_version = "base_1.0"
    model = DictToClass(
        {
            "sample_model": SDXLCreator(),
            "vae_model": create_vae_model(),  # AutoencoderKL(),
            "clip_token": create_clip_token_model(version=model_version),
            "clip_text": create_clip_text_model(version=model_version),
            "clip_vision": nn.Identity() if skip_vision else create_clip_vision_model(),
        }
    )
    return model


def create_sdxl_refiner_model():
    model_version = "refiner_1.0"
    model = DictToClass(
        {
            "sample_model": SDXLRefiner(),
            "vae_model": create_vae_model(),  # AutoencoderKL(),
            "clip_token": create_clip_token_model(version=model_version),
            "clip_text": create_clip_text_model(version=model_version),
            "clip_vision": None,
        }
    )
    return model
