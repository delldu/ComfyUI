#taken from: https://github.com/lllyasviel/ControlNet and modified
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from SDXL.util import (
    zero_module,
    timestep_embedding,
    state_dict_load,
    state_dict_filter,
)

from SDXL.attention import (
    SpatialTransformer,
)

from SDXL.unet import (
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
)

import todos
import pdb

def update_model_weight(obj, key, value):
    try:
        # obj -- ControlNet(...)
        # key -- 'time_embed.0.weight'
        attrs = key.split(".") # ['time_embed', '0', 'weight']
        for name in attrs[:-1]:
            obj = getattr(obj, name)

        setattr(obj, attrs[-1], torch.nn.Parameter(value))
    except:
        # print(f"update model weight {key} exception !!!")
        # skip output_blocks.*, out.* 
        pass


def load_ctrl_lora_weights(model, model_path="models/control-lora-canny-rank128.safetensors", unet_weight=None):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    unet_path="models/sd_xl_base_1.0.safetensors"
    unet_checkpoint = unet_path if cdir == "" else cdir + "/" + unet_path

    if os.path.exists(checkpoint) and os.path.exists(unet_checkpoint):
        print(f"Loading control-lora weight from {unet_checkpoint} and {checkpoint} ...")

        if unet_weight is None:
            unet_weight = state_dict_load(unet_path)
            unet_weight = state_dict_filter(unet_weight, ["model.diffusion_model."], remove_prefix=True)

        lora_weight = state_dict_load(model_path)

        for k in unet_weight:
            update_model_weight(model, k, unet_weight[k])

        for k in lora_weight:
            if k not in {"lora_controlnet"}:
                update_model_weight(model, k, lora_weight[k])
            else:
                pass # skip "lora_controlnet"
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}' not exist !!!")



class ControlLoraOps:
    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True,
                    device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            # in_features = 320
            # out_features = 1280
            # bias = True
            # device = None
            self.biasx = bias
            self.dtype = torch.float16
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

        def forward(self, input):
            input = input.to(self.dtype)
            if self.up is not None: # False
                return F.linear(input, self.weight.to(input.device) + (torch.mm(self.up.flatten(start_dim=1), 
                        self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), self.bias)
                # F.linear(input, self.weight.to(input.device) + (torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), self.bias)
            else:
                return F.linear(input, self.weight.to(input.device), self.bias)

        def __repr__(self):
            return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.biasx}, dtype={self.dtype})"

    class Conv2d(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = False
            self.output_padding = 0
            self.groups = groups
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None

        def forward(self, input):
            if self.up is not None:
                return F.conv2d(input, self.weight.to(input.device) + (torch.mm(self.up.flatten(start_dim=1), 
                    self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), 
                    self.bias.to(input.device), self.stride, self.padding, self.dilation, self.groups)
            else:
                return F.conv2d(input, self.weight.to(input.device), self.bias.to(input.device), self.stride, self.padding, 
                    self.dilation, self.groups)

    def conv_nd(self, dims, *args, **kwargs):
        if dims == 2:
            return self.Conv2d(*args, **kwargs)
            # return nn.Conv2d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

# control_model
class ControlNet(nn.Module):
    def __init__(
        self,
        image_size=32,
        in_channels=4,
        model_channels=320,
        hint_channels=3,
        num_res_blocks=2,
        attention_resolutions=[2, 4],
        dropout=0.0,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dims=2,
        use_fp16=True,
        num_heads=-1,
        num_head_channels=64,
        use_spatial_transformer=True,     # custom transformer support
        transformer_depth=[0, 2, 10],     # custom transformer support
        context_dim=2048,                 # custom transformer support
        use_linear_in_transformer=True,
        adm_in_channels=2816,
        transformer_depth_middle=10,
        device=None,
        operations=ControlLoraOps(),
    ):
        super(ControlNet, self).__init__()
        # operations = <comfy.controlnet.ControlLoraOps object>

        assert use_spatial_transformer == True, "use_spatial_transformer has to be true"
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
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

        assert adm_in_channels is not None
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
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels, operations=operations)])

        self.input_hint_block = TimestepEmbedSequential(
                    operations.conv_nd(dims, hint_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    operations.conv_nd(dims, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    operations.conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                    nn.SiLU(),
                    operations.conv_nd(dims, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    operations.conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                    nn.SiLU(),
                    operations.conv_nd(dims, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    operations.conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                    nn.SiLU(),
                    zero_module(operations.conv_nd(dims, 256, model_channels, 3, padding=1))
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
                        operations=operations
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels

                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, 
                            depth=transformer_depth[level], 
                            context_dim=context_dim,
                            disable_self_attn=False, 
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=False, 
                            operations=operations
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch, operations=operations))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, operations=operations)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch, operations=operations))
                ds *= 2

        num_heads = ch // num_head_channels
        dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims,
                operations=operations
            ),
            SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                            disable_self_attn=False, use_linear=use_linear_in_transformer,
                            use_checkpoint=False, operations=operations
                        ),
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims,
                operations=operations
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch, operations=operations)

        for param in self.parameters():
            param.requires_grad = False


    def make_zero_conv(self, channels, operations=None):
        return TimestepEmbedSequential(zero_module(operations.conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, y=None):
        todos.debug.output_var("ControlNet x/noise_latent_mixer", x)
        todos.debug.output_var("ControlNet hint", hint)
        todos.debug.output_var("ControlNet timesteps", timesteps)
        todos.debug.output_var("ControlNet context", context)
        todos.debug.output_var("ControlNet y", y)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        hs = []
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        # len(self.input_hint_block) -- 15
        # len(self.zero_convs) -- 9
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        todos.debug.output_var("ControlNet outs", outs)

        return outs


def canny_control_lora():
    # output: SdxlCannyLora

    model = ControlNet()
    model = model.half()
    load_ctrl_lora_weights(model, model_path="models/control-lora-canny-rank128.safetensors")
    model = model.eval()
    return model

def depth_control_lora():
    # output: SdxlDepthLora

    model = ControlNet()
    model = model.half()
    load_ctrl_lora_weights(model, model_path="models/control-lora-depth-rank128.safetensors")
    model = model.eval()
    return model

def color_control_lora():
    # output: SdxlColorLora

    model = ControlNet()
    model = model.half()
    load_ctrl_lora_weights(model, model_path="models/control-lora-recolor-rank128.safetensors")
    model = model.eval()
    return model

def sketch_control_lora():
    # output: SdxlSketchLora

    model = ControlNet()
    model = model.half()
    load_ctrl_lora_weights(model, model_path="models/control-lora-sketch-rank128.safetensors")
    model = model.eval()
    return model

if __name__ == "__main__":
    import todos
    model = canny_control_lora()
    # model = torch.jit.script(model)
    print(model)
    # todos.debug.output_weight(model.state_dict())
