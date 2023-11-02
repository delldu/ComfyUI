import torch
import torch.nn as nn
import numpy as np

from SDXL.unet import (
    Timestep,
)

from SDXL.ksampler import (
    KSampler,
)

from SDXL.noise import (
    CLIPEmbedNoiseAugmentation,
)

import todos
import pdb

class SDXLRefiner(KSampler):
    def __init__(self):
        super().__init__(version="refiner_1.0")
        self.embedder = Timestep(256)

    def encode_adm(self, cond, H, W, positive=True):
        pooled = cond['pool_encoded']

        crop_h = 0
        crop_w = 0
        aesthetic_score = 6.0 if positive else 2.5


        out = []
        out.append(self.embedder(torch.Tensor([H * 8]))) # H * 8 -- 600
        out.append(self.embedder(torch.Tensor([W * 8]))) # W * 8 -- 456
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(pooled.shape[0], 1)

        return torch.cat((pooled, flat.to(pooled.device)), dim=1)


class SDXLBase(KSampler):
    def __init__(self):
        super().__init__(version="base_1.0")
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbedNoiseAugmentation()

    def unclip_adm(self, image_embeds):
        # xxxx9999
        # image_embeds.size() -- [1, 1280]
        # tensor [image_embeds] size: [1, 1280], min: -5.467596, max: 5.339845, mean: -0.032329

        weight = 1.0
        noise_augment = 0.25 # unclip_cond["noise_augmentation"] # 0.25

        # self.noise_augmentor.max_noise_level -- 1000
        noise_level = round((self.noise_augmentor.max_noise_level - 1) * noise_augment) # ==> 0

        with torch.no_grad():
            c_adm, noise_level_emb = self.noise_augmentor(image_embeds, 
                    noise_level=torch.tensor([noise_level]).to(image_embeds.device))

        adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight

        return adm_out


    def encode_adm(self, cond, H, W, positive=True):
        if positive and 'clip_embeds' in cond: # clip_vision
            pooled = self.unclip_adm(cond['clip_embeds'])[:,:1280]
        else:
            pooled = cond['pool_encoded']

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
