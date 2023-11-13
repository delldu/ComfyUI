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
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

from SDXL.util import (
    DictToClass,

    load_model_weight,
    load_clip_vision_image,
    count_model_params,
)

from SDXL.clip_text import (
    CLIPEncoder,
)

from SDXL.noise import (
    CLIPEmbedNoiseAugmentation,
)


from typing import Dict

import todos
import pdb


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim), requires_grad=False)

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, image):
        batch_size = image.shape[0]

        patch_embeds = self.patch_embedding(image)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)

        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, image):
        hidden_states = self.embeddings(image)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs: Dict[str, torch.Tensor] = self.encoder(inputs_embeds=hidden_states, causal_attention_mask=None)

        last_hidden_state = encoder_outputs["last_hidden_state"]
        pool_encoded = last_hidden_state[:, 0, :]
        pool_encoded = self.post_layernorm(pool_encoded)

        return pool_encoded


class CLIPVisionEncoder(nn.Module):
    """
    CLIPVisionModelWithProjection
    """
    def __init__(self, preload=True):
        super().__init__()
        # come from comfy/clip_vision_config_g.json
        config = DictToClass(
            {
                "attention_dropout": 0.0,
                "dropout": 0.0,
                "hidden_act": "gelu",
                "hidden_size": 1664,
                "image_size": 224,
                "initializer_factor": 1.0,
                "initializer_range": 0.02,
                "intermediate_size": 8192,
                "layer_norm_eps": 1e-05,
                "model_type": "create_clip_vision_model",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 48,
                "atten_layer_index": 0,  # useless
                "patch_size": 14,
                "projection_dim": 1280,
                "torch_dtype": "float32",
            }
        )
        self.vision_model = CLIPVisionTransformer(config)
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        if preload:
            load_model_weight(self, model_path="models/clip_vision_g.safetensors")

        self.noise_augmentor = CLIPEmbedNoiseAugmentation()
        self.normal = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        for param in self.parameters():
            param.requires_grad = False
        self.half().eval()
        count_model_params(self)

    def get_embeds(self, image, normal_input: bool = True):
        if normal_input:
            image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
            # image normal
            image = self.normal(image)
            # tensor [image] size: [1, 3, 224, 224], min: -1.792263, max: 2.145897, mean: -0.467128

        pool_encoded = self.vision_model(image)
        image_embeds = self.visual_projection(pool_encoded)

        return image_embeds

    def forward(self, image, weight: float=1.0, noise_augment: float=0.01):
        image_embeds = self.get_embeds(image)

        # image_embeds.size() -- [1, 1280]
        # tensor [image_embeds] size: [1, 1280], min: -5.467596, max: 5.339845, mean: -0.032329

        # self.noise_augmentor.max_noise_level -- 1000
        noise_level = round((self.noise_augmentor.max_noise_level - 1) * noise_augment)  # ==> 0

        c_adm, noise_level_emb = self.noise_augmentor(
            image_embeds, noise_level=torch.tensor([noise_level]).to(image_embeds.device)
        )

        pool_encoded = torch.cat((c_adm, noise_level_emb), 1) * weight

        return pool_encoded[:, 0:1280]


def create_old_clip_vision():
    from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig

    config = CLIPVisionConfig.from_json_file("../../comfy/clip_vision_config_g.json")
    model = CLIPVisionModelWithProjection(config)
    load_model_weight(model, model_path="models/clip_vision_g.safetensors")
    model.half().eval()
    return model


def create_clip_vision_model():
    """This model only been used by sdxl base model"""

    model = CLIPVisionEncoder()
    

    return model


if __name__ == "__main__":
    import todos

    model = create_old_clip_vision()
    model.cuda()

    # todos.debug.output_weight(model.state_dict())
    image = load_clip_vision_image("../workflow/image.png")
    # todos.debug.output_var("image", image)
    # tensor [image] size: [1, 3, 224, 224], min: -1.791992, max: 2.146484, mean: -0.467529

    print("Old implement ...")
    with torch.no_grad():
        output = model(image.half().cuda(), output_hidden_states=True)
    todos.debug.output_var("output", output)

    # tensor [pool_encoded] size: [1, 1664], min: -8.867188, max: 7.449219, mean: 0.144165
    # tensor [image_embeds] size: [1, 1280], min: -6.214844, max: 4.339844, mean: -0.038574

    print("-" * 120)

    print("New implement ...")
    model = create_clip_vision_model()
    torch.save(model.state_dict(), "models/ClipVision.pth")

    model.cuda()
    with torch.no_grad():
        output = model.get_embeds(image.half().cuda(), normal_input=False)
    todos.debug.output_var("output/image_embeds", output)

    class_name = model.__class__.__name__
    model = torch.jit.script(model)
    print(f"torch.jit.script({class_name}) OK !")

