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

from SDXL.util import (
    Timestep,
)

import todos
import pdb


class SDXLRefiner(KSampler):
    def __init__(self):
        super().__init__(version="refiner_1.0")
        self.embedder = Timestep(256)

    def encode_adm(self, cond, H, W, positive=True):
        pooled = cond["pool_encoded"]

        crop_h = 0
        crop_w = 0
        aesthetic_score = 6.0 if positive else 2.5

        out = []
        out.append(self.embedder(torch.Tensor([H * 8])))  # H * 8 -- 600
        out.append(self.embedder(torch.Tensor([W * 8])))  # W * 8 -- 456
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(pooled.shape[0], 1)

        return torch.cat((pooled, flat.to(pooled.device)), dim=1)


class SDXLBase(KSampler):
    def __init__(self):
        super().__init__(version="base_1.0")
        self.embedder = Timestep(256)

    def encode_adm(self, cond, H, W, positive=True):
        pooled = cond["pool_encoded"]

        crop_w = 0
        crop_h = 0
        target_width = W * 8
        target_height = H * 8

        out = []
        out.append(self.embedder(torch.Tensor([H * 8])))
        out.append(self.embedder(torch.Tensor([W * 8])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(pooled.shape[0], 1)
        return torch.cat((pooled.to(flat.device), flat), dim=1).to(pooled.device)
