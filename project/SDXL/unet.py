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
# from abc import abstractmethod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from SDXL.attention import (
    TimestepEmbedSpatialTransformer,
)

from SDXL.util import (
    zero_module,
    timestep_embedding,
    count_model_params,
    load_unet_model_weight,
)

from typing import Dict, List

import todos
import pdb


class UNetOps:
    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
            else:
                # CrossAttention() ==> pdb.set_trace()
                self.register_parameter("bias", None)

        def forward(self, input):
            return F.linear(input, self.weight, self.bias)

        def __repr__(self):
            s = f"unet.Linear(in_features={self.in_features}, out_features={self.out_features}"
            if self.weight is not None:
                s += f", weight=Parameter({list(self.weight.size())})"
            if self.bias is not None:
                s += f", bias=Parameter({list(self.bias.size())})"
            s += ")"
            return s


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """
    def forward(self, x, emb, context):
        for layer in self:
            x = layer(x, emb, context)
        return x


class TimestepEmbedUpsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    """
    def __init__(self, channels, out_channels=None, padding=1, scale2x=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)
        self.scale2x = scale2x

    def forward(self, x, emb, context):  # x, [emb, context]
        assert x.shape[1] == self.channels
        if self.scale2x:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            x = F.interpolate(x, size=shape, mode="nearest")
        x = self.conv(x)
        return x

    def __repr__(self):
        s = f"TimestepEmbedUpsample(in_channels={self.channels}, out_channels={self.out_channels}, scale2x={self.scale2x})"
        return s


class TimestepEmbedConv2d(nn.Conv2d):
    def forward(self, input, emb, context):  # input, [emb, context]
        return self._conv_forward(input, self.weight, self.bias)


class TimestepEmbedDownsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=padding)

    def forward(self, x, emb, context):  # x, [emb, context]
        assert x.shape[1] == self.channels
        return self.op(x)

    def __repr__(self):
        s = f"TimestepEmbedDownsample(in_channels={self.channels}, out_channels={self.out_channels})"
        return s

class TimestepEmbedResBlock(nn.Module):
    """
    Shared by controlnet.py && unet.py
    """
    def __init__(self, channels, emb_channels, out_channels=None, operations=None,  # UNetOps(), ControlnetOps()
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.0),  # do not delete it !!!
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb, context):  # x, emb, [context]
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)

        return self.skip_connection(x) + h


# diffusion_model
class UNetModel(nn.Module):
    """
    sd_xl_base.yaml:
        adm_in_channels: 2816
        num_classes: sequential
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [2, 4]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        transformer_depth: [0, 2, 10]  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers
        legacy: False

    sd_xl_refine.yaml:
      params:
        adm_in_channels: 2560
        num_classes: sequential
        in_channels: 4
        out_channels: 4
        model_channels: 384
        attention_resolutions: [2, 4]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_head_channels: 64
        transformer_depth: [0, 4, 4, 0]
        context_dim: 1280
        spatial_transformer_attn_type: softmax-xformers
        legacy: False
    """

    def __init__(self, version,
        in_channels=4,
        model_channels=320, # 384
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[2, 4],
        channel_mult=(1, 2, 4), # [1, 2, 4, 4]
        num_head_channels=64,
        transformer_depth=[0, 2, 10],  # [0, 4, 4, 0]
        context_dim=2048,  # 1280
        adm_in_channels=2816, # 2560
        transformer_depth_middle=10, # 4
        operations=UNetOps(),
    ):
        super().__init__()
        self.version = version
        if version == "base_1.0":
            adm_in_channels = 2816
            model_channels = 320
            channel_mult = [1, 2, 4]
            transformer_depth = [0, 2, 10]
            context_dim = 2048
            transformer_depth_middle = 10
        else:  # refiner_1.0
            adm_in_channels = 2560
            model_channels = 384
            channel_mult = [1, 2, 4, 4]
            transformer_depth = [0, 4, 4, 0]
            context_dim = 1280
            transformer_depth_middle = 4

        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks]

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim),
        )

        self.label_emb = nn.Sequential(
            nn.Sequential(
                operations.Linear(adm_in_channels, time_embed_dim),
                nn.SiLU(),
                operations.Linear(time_embed_dim, time_embed_dim),
            )
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(TimestepEmbedConv2d(in_channels, model_channels, 3, padding=1))]
        )
        input_block_chans = [model_channels]

        ds = 1
        ch = model_channels
        for level, mult in enumerate(channel_mult):  # [1, 2, 4] or [1, 2, 4, 4]
            for nr in range(self.num_res_blocks[level]):  # self.num_res_blocks == [2, 2, 2] or [2, 2, 2, 2]
                layers = [
                    TimestepEmbedResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=mult * model_channels,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels  # 320, 640, 1280
                if ds in attention_resolutions:  # attention_resolutions -- [2, 4]
                    layers.append(
                        TimestepEmbedSpatialTransformer(
                            ch,
                            ch // num_head_channels,
                            num_head_channels,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            operations=operations,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:  # Is last channel ? channel_mult -- [1, 2, 4] or [1, 2, 4, 4]
                # self.input_blocks[3], [6] or [3, 6, 9], [0, 1, 2] ==> [3, 6, 9] == 3 * (level + 1) ?
                self.input_blocks.append(TimestepEmbedSequential(TimestepEmbedDownsample(ch, out_channels=ch)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            TimestepEmbedResBlock(ch, time_embed_dim, operations=operations),
            TimestepEmbedSpatialTransformer(
                ch,
                ch // num_head_channels,
                num_head_channels,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                operations=operations,
            ),
            TimestepEmbedResBlock(
                ch,
                time_embed_dim,
                operations=operations,
            ),
        )

        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                # self.num_res_blocks -- [2, 2, 2] or [2, 2, 2, 2]
                ich = input_block_chans.pop()
                layers = [
                    TimestepEmbedResBlock(
                        ch + ich,
                        time_embed_dim,
                        out_channels=model_channels * mult,
                        operations=operations,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        TimestepEmbedSpatialTransformer(
                            ch,
                            ch // num_head_channels,
                            num_head_channels,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            operations=operations,
                        )
                    )
                if level and i == self.num_res_blocks[level]:  # self.num_res_blocks -- [2, 2, 2] or [2, 2, 2, 2]
                    # self.output_blocks[2], [5] or [2], [5], [8], [3, 2, 1] ==> {2, 5, 8} == 2 + 3 * (3 - level)
                    if level != 1: # level == 1 means last layer for reverse channel_mult
                        layers.append(TimestepEmbedUpsample(ch, out_channels=ch))
                    else:
                        layers.append(TimestepEmbedUpsample(ch, out_channels=ch, scale2x=False))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

        if version == "base_1.0":
            load_unet_model_weight(self, model_path="models/sd_xl_base_1.0.safetensors")
        else:
            load_unet_model_weight(self, model_path="models/sd_xl_refiner_1.0.safetensors")
        for param in self.parameters():
            param.requires_grad = False
        self.half().eval()
        count_model_params(self)

    def forward(self, x, timesteps, context, y, control: Dict[str, List[torch.Tensor]]):
        # x.shape -- [2, 4, 104, 157]
        # timesteps.size() -- [2]
        # context.size() -- [2, 77, 2048]
        # y.size() -- [2, 2816]
        # control.keys() -- dict_keys(['input', 'middle', 'output'])
        # if os.environ.get('SDXL_DEBUG') is not None:
        #     todos.debug.output_var("UNetModel x/noise_latent_mixer", x)
        #     todos.debug.output_var("UNetModel timesteps", timesteps)
        #     todos.debug.output_var("UNetModel context", context)
        #     todos.debug.output_var("UNetModel y", y)
        #     todos.debug.output_var("UNetModel control['input']", control['input'])
        #     todos.debug.output_var("UNetModel control['middle']", control['middle'])
        #     todos.debug.output_var("UNetModel control['output']", control['output'])

        assert y is not None, "must specify y"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels).to(x.dtype)
        emb = self.time_embed(t_emb).to(x.dtype)

        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

        h = x
        for id, module in enumerate(self.input_blocks):  # len(self.input_blocks) -- 9 or 12
            for layer in module:
                h = layer(h, emb, context)
            hs.append(h)

        for layer in self.middle_block:
            h = layer(h, emb, context)

        if "middle" in control and len(control["middle"]) > 0:
            ctrl = control["middle"].pop()
            if ctrl is not None:
                h += ctrl

        for id, module in enumerate(self.output_blocks):  # len(self.output_blocks) -- 9 or 12
            hsp = hs.pop()
            if "output" in control and len(control["output"]) > 0:
                ctrl = control["output"].pop()
                if ctrl is not None:
                    hsp += ctrl

            h = torch.cat([h, hsp], dim=1)
            del hsp

            for layer in module:
                h = layer(h, emb, context)

            # xxxx8888
            # if len(hs) > 0:
            #     B, C, H, W = hs[-1].size()
            #     h = F.interpolate(h, size=(H, W), mode="nearest")  # Restore to old size for upsample 2x

        output = self.out(h)
        # if os.environ.get('SDXL_DEBUG') is not None:
        #     todos.debug.output_var("UNetModel output", output)

        return output


def create_base_unet_model():
    model = UNetModel(version="base_1.0")
    return model


def create_refiner_unet_model():
    model = UNetModel(version="refiner_1.0")
    return model


if __name__ == "__main__":
    model = create_base_unet_model()
    class_name = model.__class__.__name__
    model = torch.jit.script(model)
    print(f"torch.jit.script({class_name}) OK !")

    model = create_refiner_unet_model()
    class_name = model.__class__.__name__
    model = torch.jit.script(model)
    print(f"torch.jit.script({class_name}) OK !")

    # # todos.debug.output_weight(model.state_dict())
