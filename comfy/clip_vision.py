from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig, CLIPImageProcessor, modeling_utils
from .utils import load_torch_file, transformers_convert
import os
import torch
import contextlib

import comfy.ops
import comfy.model_patcher
import comfy.model_management
import todos
import pdb

class ClipVisionModel():
    def __init__(self, json_config):
        # json_config -- comfy/clip_vision_config_g.json

        config = CLIPVisionConfig.from_json_file(json_config)
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.float32
        if comfy.model_management.should_use_fp16(self.load_device, prioritize_performance=False):
            self.dtype = torch.float16

        # offload_device -- (device(type='cpu'),)
        with comfy.ops.use_comfy_ops(offload_device, self.dtype):
            with modeling_utils.no_init_weights():
                self.model = CLIPVisionModelWithProjection(config)
        self.model.to(self.dtype)

        self.patcher = comfy.model_patcher.ModelPatcher(self.model, load_device=self.load_device, offload_device=offload_device)
        self.processor = CLIPImageProcessor(crop_size=224,
                                            do_center_crop=True,
                                            do_convert_rgb=True,
                                            do_normalize=True,
                                            do_resize=True,
                                            image_mean=[ 0.48145466,0.4578275,0.40821073],
                                            image_std=[0.26862954,0.26130258,0.27577711],
                                            resample=3, #bicubic
                                            size=224)

    def load_sd(self, sd):
        # return self.model.load_state_dict(sd, strict=False)
        return self.model.load_state_dict(sd, strict=True)

    def encode_image(self, image):
        img = torch.clip((255. * image), 0, 255).round().int()
        img = list(map(lambda a: a, img))
        inputs = self.processor(images=img, return_tensors="pt")
        comfy.model_management.load_model_gpu(self.patcher)
        pixel_values = inputs['pixel_values'].to(self.load_device)
        # tensor [pixel_values] size: [1, 3, 224, 224], min: -1.792263, max: 2.145897, mean: -0.467128

        if self.dtype != torch.float32: # True for self.dtype -- torch.float16
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, b: contextlib.nullcontext(a)

        # from SDXL.util import load_clip_vision_image
        # pixel_values = load_clip_vision_image("project/workflow/image.png").cuda()

        with precision_scope(comfy.model_management.get_autocast_device(self.load_device), torch.float32):
            # self.model -- CLIPVisionModelWithProjection
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        # outputs.keys() -- ['image_embeds', 'last_hidden_state', 'hidden_states']
        for k in outputs:
            t = outputs[k]
            if t is not None:
                if k == 'hidden_states':
                    # outputs["penultimate_hidden_states"] = t[-2].cpu()
                    outputs["hidden_states"] = None
                else:
                    outputs[k] = t.cpu()

        return outputs

def convert_to_transformers(sd, prefix):
    sd_k = sd.keys()
    if "{}transformer.resblocks.0.attn.in_proj_weight".format(prefix) in sd_k:
        keys_to_replace = {
            "{}class_embedding".format(prefix): "vision_model.embeddings.class_embedding",
            "{}conv1.weight".format(prefix): "vision_model.embeddings.patch_embedding.weight",
            "{}positional_embedding".format(prefix): "vision_model.embeddings.position_embedding.weight",
            "{}ln_post.bias".format(prefix): "vision_model.post_layernorm.bias",
            "{}ln_post.weight".format(prefix): "vision_model.post_layernorm.weight",
            "{}ln_pre.bias".format(prefix): "vision_model.pre_layrnorm.bias",
            "{}ln_pre.weight".format(prefix): "vision_model.pre_layrnorm.weight",
        }

        for x in keys_to_replace:
            if x in sd_k:
                sd[keys_to_replace[x]] = sd.pop(x)

        if "{}proj".format(prefix) in sd_k:
            sd['visual_projection.weight'] = sd.pop("{}proj".format(prefix)).transpose(0, 1)

        pdb.set_trace()
        sd = transformers_convert(sd, prefix, "vision_model.", 48)
    return sd

def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    # prefix = ''
    # convert_keys = False
    if convert_keys:
        sd = convert_to_transformers(sd, prefix)
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd: # True
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_g.json")
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_h.json")
    else:
        json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_vision_config_vitl.json")
    clip = ClipVisionModel(json_config)
    m, u = clip.load_sd(sd)
    if len(m) > 0:
        print("missing clip vision:", m)

    return clip

def load(ckpt_path):
    # ckpt_path -- models/clip_vision/clip_vision_g.safetensors
    sd = load_torch_file(ckpt_path)

    if "visual.transformer.resblocks.0.attn.in_proj_weight" in sd: # False
        return load_clipvision_from_sd(sd, prefix="visual.", convert_keys=True)
    else:
        return load_clipvision_from_sd(sd)
