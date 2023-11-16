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

