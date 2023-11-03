import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
import pdb

# first_stage_model, only for test network struct !!!
class AutoencoderKL(nn.Module):
    """
    sdxl_base.yaml/sdxl_refine.yaml
        first_stage_config:
          target: sgm.models.autoencoder.AutoencoderKLInferenceWrapper
          params:
            embed_dim: 4
            monitor: val/rec_loss
            ddconfig:
              attn_type: vanilla-xformers
              double_z: true
              z_channels: 4
              resolution: 256
              in_channels: 3
              out_ch: 3
              ch: 128
              ch_mult: [1, 2, 4, 4]
              num_res_blocks: 2
              attn_resolutions: []
              dropout: 0.0
            lossconfig:
              target: torch.nn.Identity
    """

    def __init__(self, version, embed_dim=4, z_channels=4):
        super(AutoencoderKL, self).__init__()
        self.version = version
        self.encoder = Encoder()

        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)

        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder()  # model size 190 M

        for param in self.parameters():
            param.requires_grad = False

        # https://huggingface.co/stabilityai/sdxl-vae, better performance !!!
        # load_vae_model_weight(self, model_path="models/sdxl_vae.safetensors")

    def forward(self, x):
        x = self.encoder(x)
        return self.decode(x)

    def encode(self, x):
        # tensor [x] size: [1, 3, 832, 1256] , min: -1.0 , max: 1.0 mean: 0.5469990968704224
        h = self.encoder(x)
        moments = self.quant_conv(h)

        # Create DiagonalGaussianDistribution
        meanvar, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        stdvar = torch.exp(0.5 * logvar)
        output = meanvar + stdvar * torch.randn(meanvar.shape).to(device=x.device)

        return output


    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

# create_vae_encode_model
class VAEEncode(nn.Module):
    def __init__(self, version, embed_dim=4, z_channels=4):
        super().__init__()

        self.version = version
        self.encoder = Encoder()
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = 2.0 * x - 1.0 # convert x from [0.0, 1.0] to [-1.0, 1.0]
        h = self.encoder(x)
        moments = self.quant_conv(h)

        # Create DiagonalGaussianDistribution
        meanvar, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        stdvar = torch.exp(0.5 * logvar)
        output = meanvar + stdvar * torch.randn(meanvar.shape).to(device=x.device)

        return output

# create_vae_decode_model
class VAEDecode(nn.Module):
    def __init__(self, version, embed_dim=4, z_channels=4):
        super().__init__()
        # ddconfig = {
        #     "z_channels": 4,
        #     "resolution": 256,
        #     "in_channels": 3,
        #     "out_ch": 3,
        #     "ch": 128,
        #     "ch_mult": [1, 2, 4, 4],
        #     "num_res_blocks": 2,
        #     "dropout": 0.0,
        # }
        self.version = version
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        # self.decoder = Decoder(**ddconfig)  # model size 190 M
        self.decoder = Decoder()  # model size 190 M

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # tensor [dec] size: [1, 3, 600, 456], min: -1.524532, max: 1.899036, mean: 0.059854
        # tensor [zout] size: [1, 3, 600, 456], min: -1.649717, max: 1.515323, mean: 0.132423

        out = ((dec + 1.0)/2.0).clamp(0.0, 1.0)

        return out

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:  # To support torch.jit.script
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttnBlock, self).__init__()
        # self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    def __init__(self, 
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0,
        in_channels=3,
        resolution=256,
        z_channels=4,
    ):
        super().__init__()

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution # 256
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions): # 4
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            else: # support torch.jit.script
                down.downsample = nn.Identity()

            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        h = self.conv_in(x)

        # Support torch.jit.script
        # for i_level in range(self.num_resolutions): # self.num_resolutions -- 4
        #     for i_block in range(self.num_res_blocks): # self.num_res_blocks -- 2
        #         h = self.down[i_level].block[i_block](h)
        #     if i_level != self.num_resolutions - 1:
        #         h = self.down[i_level].downsample(h)
        for i_level, layer in enumerate(self.down):
            for i_block, block in enumerate(layer.block):
                h = block(h)
            if i_level != self.num_resolutions - 1:
                h = layer.downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0,
        in_channels=3,
        resolution=256,
        z_channels=4,
    ):
        super(Decoder, self).__init__()

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            else: # Support torch.jit.script
                up.upsample = nn.Identity()                
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # xxxx9999
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # xxxx8888
        # for i in range(self.num_resolutions):
        #     h = self.up_layer(self.num_resolutions - i - 1, h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def up_layer(self, i:int, h):
        '''ugly for torch.jit.script no reversed(), oh oh oh !!!'''
        for i_level, layer in enumerate(self.up):
            if i_level == i:
                for i_block, block in enumerate(layer.block):
                    h = block(h)
                h = layer.upsample(h)
        return h


def create_vae_model():
    model = AutoencoderKL(version="base_1.0")
    model = model.half()
    model = model.eval()

    return model


def create_vae_encode_model():
    # output: SdxlVaeEncode

    model = VAEEncode(version="base_1.0")
    # model = model.half()
    model = model.eval()
    # model = model.cuda()

    return model


def create_vae_decode_model():
    # output: SdxlVaeDecode

    model = VAEDecode(version="base_1.0")
    # model = model.half()
    model = model.eval()
    # model = model.cuda()

    return model


if __name__ == "__main__":
    model = create_vae_model()
    # model = create_vae_encode_model()
    # model = create_vae_decode_model()

    # model = torch.jit.script(model)
    print(model)
