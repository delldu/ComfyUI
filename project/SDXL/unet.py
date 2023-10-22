from abc import abstractmethod
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SDXL.attention import (
    SpatialTransformer,
)

import SDXL.util
from SDXL.util import (
    load_diffusion_model_weight,
    timestep_embedding,
    zero_module,
    avg_pool_nd,
)

import pdb


def exists(val):
    return val is not None


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, output_shape=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
        return x

def forward_timestep_embed(ts, x, emb, context=None, output_shape=None):
    for layer in ts:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context)
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=None, device=None, operations=SDXL.util):
        super(Upsample, self).__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            # ==> pdb.set_trace()
            self.conv = operations.conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, dtype=dtype, device=device)

    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]

        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=None, device=None, operations=SDXL.util):
        super(Downsample, self).__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            # ==> pdb.set_trace()
            self.op = operations.conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, dtype=dtype, device=device
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        dtype=None,
        device=None,
        operations=SDXL.util
    ):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=dtype, device=device),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(emb_channels, self.out_channels, dtype=dtype, device=device),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                operations.conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, dtype=dtype, device=device)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, 3, padding=1, dtype=dtype, device=device
            )
        else:
            self.skip_connection = operations.conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)


    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)

        return self.skip_connection(x) + h

class Timestep(nn.Module):
    def __init__(self, dim):
        super(Timestep, self).__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)

    def __repr__(self):
        return f"Timestep({self.dim})"


# diffusion_model
class UNetModel(nn.Module):
    """
    sd_xl_base.yaml:    
        adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: False
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [2, 4]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_spatial_transformer: True
        use_linear_in_transformer: True
        transformer_depth: [0, 2, 10]  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers
        legacy: False

    sd_xl_refine.yaml:    
      params:
        adm_in_channels: 2560
        num_classes: sequential
        use_checkpoint: False
        in_channels: 4
        out_channels: 4
        model_channels: 384
        attention_resolutions: [2, 4]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_head_channels: 64
        use_spatial_transformer: True
        use_linear_in_transformer: True
        transformer_depth: [0, 4, 4, 0]
        context_dim: 1280
        spatial_transformer_attn_type: softmax-xformers
        legacy: False
    """

    def __init__(
        self,
        version="base_1.0",
        image_size=32,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[2, 4],
        dropout=0,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dims=2,
        use_fp16=True,
        num_heads=-1,
        num_head_channels=64,
        use_spatial_transformer=True,    # custom transformer support
        transformer_depth=[0, 2, 10],    # custom transformer support
        context_dim=2048,                # custom transformer support
        use_linear_in_transformer=True,
        adm_in_channels=2816,
        transformer_depth_middle=10,
        device=None,
        operations=SDXL.util,
    ):
        super(UNetModel, self).__init__()
        if version == "base_1.0":
            adm_in_channels = 2816
            model_channels = 320
            channel_mult = [1, 2, 4]
            transformer_depth = [1, 2, 10]
            context_dim = 2048
            transformer_depth_middle=10
        else: # refiner_1.0
            adm_in_channels = 2560
            model_channels = 384
            channel_mult = [1, 2, 4, 4]
            transformer_depth = [0, 4, 4, 0]
            context_dim = 1280
            transformer_depth_middle = 4

        assert use_spatial_transformer == True, "use_spatial_transformer has to be true"
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'

        self.context_dim = context_dim

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None:
            transformer_depth_middle =  transformer_depth[-1]
        self.num_res_blocks = len(channel_mult) * [num_res_blocks]

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        self.label_emb = nn.Sequential(
            nn.Sequential(
                operations.Linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                nn.SiLU(),
                operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
            )
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    operations.conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels

                    layers.append(SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim,
                            disable_self_attn=False, use_linear=use_linear_in_transformer,
                            dtype=self.dtype, device=device, operations=operations
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        num_heads = ch // num_head_channels
        dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims,
                dtype=self.dtype,
                device=device,
                operations=operations
            ),
            SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                            disable_self_attn=False, use_linear=use_linear_in_transformer,
                            dtype=self.dtype, device=device, operations=operations
                        ),
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims,
                dtype=self.dtype,
                device=device,
                operations=operations
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        dtype=self.dtype,
                        device=device,
                        operations=operations
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels

                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim,
                            disable_self_attn=False, use_linear=use_linear_in_transformer,
                            dtype=self.dtype, device=device, operations=operations
                        )
                    )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch, dtype=self.dtype, device=device),
            nn.SiLU(),
            zero_module(operations.conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=self.dtype, device=device)),
        )

        if version == "base_1.0":
            load_diffusion_model_weight(self, model_path="models/sd_xl_base_1.0.safetensors")
        else:
            load_diffusion_model_weight(self, model_path="models/sd_xl_refiner_1.0.safetensors")

    def forward(self, x, timesteps=None, context=None, y=None, control=None):
        # x.shape -- [2, 4, 104, 157]
        # timesteps.size() -- [2]
        # context.size() -- [2, 77, 2048]
        # y.size() -- [2, 2816]
        # control.keys() -- dict_keys(['input', 'middle', 'output'])

        assert (y is not None), "must specify y"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for id, module in enumerate(self.input_blocks):
            h = forward_timestep_embed(module, h, emb, context)
            hs.append(h)
        h = forward_timestep_embed(self.middle_block, h, emb, context)
        if control is not None and 'middle' in control and len(control['middle']) > 0:
            # CannyImage ==> Here, ClipVision NOT
            ctrl = control['middle'].pop()
            if ctrl is not None:
                h += ctrl

        for id, module in enumerate(self.output_blocks):
            hsp = hs.pop()
            if control is not None and 'output' in control and len(control['output']) > 0:
                # CannyImage ==> Here, ClipVision NOT
                ctrl = control['output'].pop()
                if ctrl is not None:
                    hsp += ctrl

            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, output_shape)
        h = h.type(x.dtype)
        return self.out(h)


def sdxl_unet_model():
    # output: SdxlUnetModel

    model = UNetModel(version="base_1.0")
    model = model.eval()
    return model    


def refiner_unet_model():
    # output: RefinerUnetModel

    model = UNetModel(version="refiner_1.0")
    model = model.eval()
    return model    

if __name__ == "__main__":
    import todos

    model = sdxl_unet_model()
    # model = torch.jit.script(model)
    # print(model)

    # model = refiner_unet_model()
    # model = torch.jit.script(model)
    # print(model)


    todos.debug.output_weight(model.state_dict())
