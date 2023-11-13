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
# taken from: https://github.com/lllyasviel/ControlNet and modified
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from SDXL.util import (
    zero_module,
    timestep_embedding,
    state_dict_load,
    state_dict_filter,
    load_model_weight,
    count_model_params,
)

from SDXL.attention import (
    TimestepEmbedSpatialTransformer,
)

from SDXL.unet import (
    TimestepEmbedSequential,
)

from typing import Dict, List

import todos
import pdb


def update_lora_weight(obj, key, value):
    try:
        # obj -- ControlNet(...)
        # key -- 'time_embed.0.weight'
        attrs = key.split(".")  # ['time_embed', '0', 'weight']
        for name in attrs[:-1]:
            obj = getattr(obj, name)

        setattr(obj, attrs[-1], nn.Parameter(value))
    except:
        # print(f"update model weight {key} exception !!!")
        # skip output_blocks.*, out.*
        pass


def load_ctrl_lora_weight(model, model_path="models/control-lora-canny-rank128.safetensors", unet_weight=None):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    unet_path = "models/sd_xl_base_1.0.safetensors"
    unet_checkpoint = unet_path if cdir == "" else cdir + "/" + unet_path

    if os.path.exists(checkpoint) and os.path.exists(unet_checkpoint):
        print(f"Loading control-lora weight from {unet_checkpoint} and {checkpoint} ...")

        if unet_weight is None:
            unet_weight = state_dict_load(unet_path)
            unet_weight = state_dict_filter(unet_weight, ["model.diffusion_model."], remove_prefix=True)

        lora_weight = state_dict_load(model_path)

        for k in unet_weight:
            update_lora_weight(model, k, unet_weight[k])

        for k in lora_weight:
            if k not in {"lora_controlnet"}:
                update_lora_weight(model, k, lora_weight[k])
            else:
                pass  # skip "lora_controlnet"
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}' not exist !!!")


class ControlLoraOps:
    class Linear(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.biasx = bias
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
            else:
                self.bias = None
            self.up = nn.Parameter(torch.zeros(out_features, 128), requires_grad=False)
            self.down = nn.Parameter(torch.zeros(128, in_features), requires_grad=False)

            # lora.Linear(in_features=320, out_features=1280, weight=Parameter([1280, 320]), bias=Parameter([1280]), 
            #     up=Parameter([1280, 128(r)]), down=Parameter([128(r), 320]))
            # lora.Linear(in_features=1280, out_features=640, weight=Parameter([640, 1280]), bias=Parameter([640]), 
            #     up=Parameter([640, 128(r)]), down=Parameter([128(r), 1280]))


        def forward(self, input):
            return F.linear(input, self.weight + torch.mm(self.up, self.down), self.bias)

        def __repr__(self):
            s = f"lora.Linear(in_features={self.in_features}, out_features={self.out_features}"
            s += f", weight=Parameter({list(self.weight.size())})"
            if self.bias is not None:
                s += f", bias=Parameter({list(self.bias.size())})"
            if self.up is not None:
                s += f", up=Parameter({list(self.up.size())})"
            if self.down is not None:
                s += f", down=Parameter({list(self.down.size())})"
            s += ")"
            return s

    class Conv2d(nn.Module):
        def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            rank=128,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.rank = rank
            self.weight = nn.Parameter(
                torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=False
            )
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            if rank > 0:
                self.up = nn.Parameter(torch.zeros(out_channels, rank, 1, 1), requires_grad=False)
                self.down = nn.Parameter(
                    torch.zeros(rank, in_channels, kernel_size, kernel_size), requires_grad=False
                )
            else:
                self.up = None
                self.down = None
            # lora.Conv2d(in_channels=4, out_channels=320, kernel_size=3, stride=1, weight=Parameter([320, 4, 3, 3]), 
            #     bias=Parameter([320]), up=Parameter([320, 4(r), 1, 1]), down=Parameter([4(r), 4, 3, 3]))
            # lora.Conv2d(in_channels=320, out_channels=640, kernel_size=3, stride=1, weight=Parameter([640, 320, 3, 3]), 
            #     bias=Parameter([640]), up=Parameter([640, 128(r), 1, 1]), down=Parameter([128(r), 320, 3, 3]))


        def forward(self, input, emb=None, context=None):  # x, [emb, context]
            if self.rank > 0:
                return F.conv2d(
                    input,
                    self.weight
                    + torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1)).reshape(self.weight.shape),
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            else:
                return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        def __repr__(self):
            s = f"lora.Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}"
            s += f", kernel_size={self.kernel_size}, stride={self.stride}"
            s += f", weight=Parameter({list(self.weight.size())})"
            s += f", bias=Parameter({list(self.bias.size())})"
            if self.up is not None:
                s += f", up=Parameter({list(self.up.size())})"
            if self.down is not None:
                s += f", down=Parameter({list(self.down.size())})"
            s += ")"
            return s



class TimestepEmbedResBlock(nn.Module):
    """
    Shared by controlnet.py && unet.py
    """
    def __init__(self, channels, emb_channels, out_channels, use_conv=False, operations=None,  # UNetOps(), ControlnetOps()
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            operations.Conv2d(channels, self.out_channels, 3, padding=1, rank=128),
        )
        # (in_layers): Sequential(
        #     (0): GroupNorm(32, 1280, eps=1e-05, affine=True)
        #     (1): SiLU()
        #     (2): lora.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, stride=1, 
        #         weight=Parameter([1280, 1280, 3, 3]), bias=Parameter([1280]), 
        #         up=Parameter([1280, 128, 1, 1]), down=Parameter([128, 1280, 3, 3]))
        # )
        # (in_layers): Sequential(
        #   (0): GroupNorm(32, 640, eps=1e-05, affine=True)
        #   (1): SiLU()
        #   (2): lora.Conv2d(in_channels=640, out_channels=1280, kernel_size=3, stride=1, 
        #     weight=Parameter([1280, 640, 3, 3]), bias=Parameter([1280]), 
        #     up=Parameter([1280, 128, 1, 1]), down=Parameter([128, 640, 3, 3]))
        # )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(emb_channels, self.out_channels, bias=True),
        )
        # (emb_layers): Sequential(
        #     (0): SiLU()
        #     (1): lora.Linear(in_features=1280, out_features=1280, weight=Parameter([1280, 1280]), bias=Parameter([1280]), 
        #         up=Parameter([1280, 128]), down=Parameter([128, 1280]))
        # )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.0),  # do not delete it !!!
            zero_module(operations.Conv2d(self.out_channels, self.out_channels, 3, padding=1, rank=128)),
        )
        # (out_layers): Sequential(
        #     (0): GroupNorm(32, 1280, eps=1e-05, affine=True)
        #     (1): SiLU()
        #     (2): Dropout(p=0.0, inplace=False)
        #     (3): lora.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, stride=1, 
        #         weight=Parameter([1280, 1280, 3, 3]), bias=Parameter([1280]), 
        #         up=Parameter([1280, 128, 1, 1]), down=Parameter([128, 1280, 3, 3]))
        # )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            pdb.set_trace()
            self.skip_connection = operations.Conv2d(channels, self.out_channels, 3)
        else:
            # self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
            # xxxx_dddd
            self.skip_connection = operations.Conv2d(channels, self.out_channels, 1)
        # (skip_connection): lora.Conv2d(in_channels=320, out_channels=640, kernel_size=1, stride=1, 
        #     weight=Parameter([640, 320, 1, 1]), bias=Parameter([640]), 
        #     up=Parameter([640, 128, 1, 1]), down=Parameter([128, 320, 1, 1]))

    def forward(self, x, emb, context):  # x, emb, [context]
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)

        return self.skip_connection(x) + h


class TimestepEmbedDownsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels, out_channels=None, padding=1, operations=ControlLoraOps()):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        # xxxx_dddd
        self.op = operations.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=padding)

    def forward(self, x, emb, context):  # x, [emb, context]
        assert x.shape[1] == self.channels
        return self.op(x, emb, context)


class TimestepEmbedSiLU(nn.SiLU):
    def forward(self, x, emb, context):  # x, [emb, context]
        return F.silu(x, inplace=self.inplace)


class ControlNot(nn.Module):
    def __init__(self, preload=True):
        super().__init__()

    def forward(self, x, hint, timesteps, context, y) -> List[torch.Tensor]:
        return []

# control_model
class ControlNet(nn.Module):
    def __init__(self,
        in_channels=4,
        model_channels=320,
        hint_channels=3,
        num_res_blocks=2,
        attention_resolutions=[2, 4],
        channel_mult=(1, 2, 4),
        num_head_channels=64,
        transformer_depth=[0, 2, 10],
        context_dim=2048,
        adm_in_channels=2816,
        transformer_depth_middle=10,
        operations=ControlLoraOps(),
        preload=True,
    ):
        super().__init__()

        self.model_channels = model_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks]

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim),  # 320, 1280, up=1280, 128
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim),  # 1280, 1280, up=1280, 128
        )

        assert adm_in_channels is not None
        self.label_emb = nn.Sequential(
            nn.Sequential(
                operations.Linear(adm_in_channels, time_embed_dim),
                nn.SiLU(),
                operations.Linear(time_embed_dim, time_embed_dim),
            )
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(operations.Conv2d(in_channels, model_channels, 3, padding=1, rank=4))]
        )

        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            operations.Conv2d(hint_channels, 16, 3, padding=1, rank=0), # [16, 3, 3, 3]
            TimestepEmbedSiLU(),
            operations.Conv2d(16, 16, 3, padding=1, rank=0),
            TimestepEmbedSiLU(),
            operations.Conv2d(16, 32, 3, padding=1, stride=2, rank=0), # [32, 16, 3, 3]
            TimestepEmbedSiLU(),
            operations.Conv2d(32, 32, 3, padding=1, rank=0),
            TimestepEmbedSiLU(),
            operations.Conv2d(32, 96, 3, padding=1, stride=2, rank=0), #[96, 32, 3, 3]
            TimestepEmbedSiLU(),
            operations.Conv2d(96, 96, 3, padding=1, rank=0),
            TimestepEmbedSiLU(),
            operations.Conv2d(96, 256, 3, padding=1, stride=2, rank=0), # [256, 96, 3, 3]
            TimestepEmbedSiLU(),
            zero_module(operations.Conv2d(256, model_channels, 3, padding=1, rank=0)), # [320, 256, 3, 3]
        )

        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult): # channel_mult=(1, 2, 4)
            for nr in range(self.num_res_blocks[level]): # [2, 2, 2]
                layers = [
                    TimestepEmbedResBlock(
                        ch,
                        time_embed_dim, # 1280
                        out_channels=mult * model_channels, # 320, 640, 1280
                        operations=operations,
                    )
                ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels

                    layers.append(
                        TimestepEmbedSpatialTransformer(
                            ch,
                            ch // num_head_channels, # num_heads,
                            num_head_channels, # dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            operations=operations,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    TimestepEmbedDownsample(ch, ch)))

                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2


        # num_heads = ch // num_head_channels
        # dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            TimestepEmbedResBlock(ch, time_embed_dim, ch, operations=operations),

            TimestepEmbedSpatialTransformer(
                ch,
                ch // num_head_channels, #num_heads,
                num_head_channels, # dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                operations=operations,
            ),
            TimestepEmbedResBlock(ch, time_embed_dim, ch, operations=operations),
        )
        self.middle_block_out = self.make_zero_conv(ch)

        pdb.set_trace()

        if preload:
            load_ctrl_lora_weight(self, model_path="models/control-lora-canny-rank128.safetensors")
        for param in self.parameters():
            param.requires_grad = False
        self.half().eval()
        count_model_params(self)

    def make_zero_conv(self, channels, operations=ControlLoraOps()):
        return TimestepEmbedSequential(
            zero_module(operations.Conv2d(channels, channels, 1, padding=0, rank=0))
        )

    def forward(self, x, hint, timesteps, context, y) -> List[torch.Tensor]:
        # if os.environ.get('SDXL_DEBUG') is not None:
        #     todos.debug.output_var("ControlNet x/noise_latent_mixer", x)
        #     todos.debug.output_var("ControlNet hint", hint)
        #     todos.debug.output_var("ControlNet timesteps", timesteps)
        #     todos.debug.output_var("ControlNet context", context)
        #     todos.debug.output_var("ControlNet y", y)

        t_emb = timestep_embedding(timesteps, self.model_channels).to(x.dtype)
        emb = self.time_embed(t_emb).to(x.dtype)
        guided_hint = self.input_hint_block(hint, emb, context)

        output: List[torch.Tensor] = []

        hs = []
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

        h = x
        # len(self.input_hint_block) -- 15
        # len(self.zero_convs) -- 9
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            output.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        output.append(self.middle_block_out(h, emb, context))

        # if os.environ.get('SDXL_DEBUG') is not None:
        #     todos.debug.output_var("ControlNet output", output)

        return output


def create_canny_control_lora():
    model = ControlNet()
    return model


def create_depth_control_lora():
    model = ControlNet()
    load_ctrl_lora_weight(model, model_path="models/control-lora-depth-rank128.safetensors")
    return model


def create_color_control_lora():
    model = ControlNet()
    load_ctrl_lora_weight(model, model_path="models/control-lora-recolor-rank128.safetensors")
    return model


def create_sketch_control_lora():
    model = ControlNet()
    #     load_ctrl_lora_weight(model, model_path="models/control-lora-sketch-rank128-metadata.safetensors")

    return model


if __name__ == "__main__":
    model = create_canny_control_lora()
    pdb.set_trace()

    class_name = model.__class__.__name__
    model = torch.jit.script(model)
    print(f"torch.jit.script({class_name}) OK !")

    # todos.debug.output_weight(model.state_dict())

    # model = create_depth_control_lora()
    # model = create_color_control_lora()
    # # model = create_sketch_control_lora()


    # size mismatch for lora_model.time_embed.0.up: copying a param with shape torch.Size([1280, 128]) from checkpoint, 
    #     the shape in current model is torch.Size([128, 1280]).

    # size mismatch for lora_model.time_embed.0.down: copying a param with shape torch.Size([128, 320]) from checkpoint, 
    #     the shape in current model is torch.Size([1280, 128]).


    # size mismatch for lora_model.time_embed.2.up: copying a param with shape torch.Size([1280, 128]) from checkpoint, 
    #     the shape in current model is torch.Size([128, 1280]).
    # size mismatch for lora_model.time_embed.2.down: copying a param with shape torch.Size([128, 1280]) from checkpoint, 
    #     the shape in current model is torch.Size([1280, 128]).

    # Sequential(
    #   (0): lora.Linear(in_features=320, out_features=1280, weight=Parameter([1280, 320]), bias=Parameter([1280]), 
    #     up=Parameter([1280, 128]), down=Parameter([1280, 128]))
    #   (1): SiLU()
    #   (2): lora.Linear(in_features=1280, out_features=1280, weight=Parameter([1280, 1280]), bias=Parameter([1280]), 
    #     up=Parameter([1280, 128]), down=Parameter([1280, 128]))
    # )
