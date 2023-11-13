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
import torch
import torch.nn as nn
import numpy as np

from SDXL.ksampler import (
    KSampler,
)
from SDXL.clip_text import (
    CreatorCLIPTextEncoder,
    RefinerCLIPTextEncoder,
)
from SDXL.vae import (
    AutoEncoder,
)

from SDXL.util import (
    Timestep,
    load_model_weight,
    count_model_params,
)
from typing import Dict

import todos
import pdb


class ImageCreator(KSampler):
    def __init__(self, preload=False):
        super().__init__(version="base_1.0", preload=preload)
        self.embedder = Timestep(256)
        # unet_model, lora_model
        self.clip_model = CreatorCLIPTextEncoder(preload=preload)
        self.vae_model = AutoEncoder(preload=preload)

        if not preload: # Fast mode
            # self.lora_model.load_state_dict(torch.load("/tmp/lora.pth"))
            load_model_weight(self, "models/ImageCreator.pth")
        for param in self.parameters():
            param.requires_grad = False
        self.half().eval()
        count_model_params(self)

    def encode_adm(self, cond: Dict[str, torch.Tensor], H:int, W:int, positive:bool=True):
        pooled = cond["pool_encoded"]

        crop_w = 0
        crop_h = 0
        target_width = W * 8
        target_height = H * 8

        out = []
        out.append(self.embedder(torch.tensor([H * 8])))
        out.append(self.embedder(torch.tensor([W * 8])))
        out.append(self.embedder(torch.tensor([crop_h])))
        out.append(self.embedder(torch.tensor([crop_w])))
        out.append(self.embedder(torch.tensor([target_height])))
        out.append(self.embedder(torch.tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(pooled.shape[0], 1)
        return torch.cat((pooled.to(flat.device), flat), dim=1).to(pooled.device)


class ImageRefiner(KSampler):
    def __init__(self, preload=False):
        super().__init__(version="refiner_1.0", preload=preload)
        self.embedder = Timestep(256)
        # unet_model, lora_model(x)
        self.clip_model = RefinerCLIPTextEncoder(preload=preload)
        self.vae_model = AutoEncoder(preload=preload)

        if not preload: # Fast mode
            load_model_weight(self, "models/ImageRefiner.pth")
        for param in self.parameters():
            param.requires_grad = False
        self.half().eval()
        count_model_params(self)

    def encode_adm(self, cond: Dict[str, torch.Tensor], H:int, W:int, positive:bool=True):
        pooled = cond["pool_encoded"]

        crop_h = 0
        crop_w = 0
        aesthetic_score = 6.0 if positive else 2.5

        out = []
        out.append(self.embedder(torch.tensor([H * 8])))  # H * 8 -- 600
        out.append(self.embedder(torch.tensor([W * 8])))  # W * 8 -- 456
        out.append(self.embedder(torch.tensor([crop_h])))
        out.append(self.embedder(torch.tensor([crop_w])))
        out.append(self.embedder(torch.tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(pooled.shape[0], 1)

        return torch.cat((pooled, flat.to(pooled.device)), dim=1)


if __name__ == "__main__":
    model = ImageCreator(preload=False)
    # torch.save(model.state_dict(), "models/ImageCreator.pth")

    # class_name = model.__class__.__name__
    # model = torch.jit.script(model)
    # print(f"torch.jit.script({class_name}) OK !")


    # model = ImageRefiner(preload=False)
    # torch.save(model.state_dict(), "models/ImageRefiner.pth")

    # class_name = model.__class__.__name__
    # model = torch.jit.script(model)
    # print(f"torch.jit.script({class_name}) OK !")
