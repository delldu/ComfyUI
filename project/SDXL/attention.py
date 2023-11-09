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
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from typing import Optional
import pdb

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, operations=None): # UNetOps(), ControlnetOps()
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dtype=None, operations=None): # UNetOps(), ControlnetOps()

        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = GEGLU(dim, inner_dim, operations=operations)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(0.0),
            operations.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, operations=None): # UNetOps(), ControlnetOps()
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
                        operations.Linear(inner_dim, query_dim),
                        nn.Dropout(0.0),
                    )

        self.BxHxNxD_BxNxHD = Rearrange('b h n d -> b n (h d)')
        self.BxNxHxD_BHxNxD = Rearrange('b n h d -> (b h) n d')
        
    def forward(self, x, context: Optional[torch.Tensor]):
        q = self.to_q(x)
        # context = default(context, x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        # q, k, v = map(
        #     lambda t: t.view(b, -1, self.heads, self.dim_head).transpose(1, 2),
        #     (q, k, v),
        # )
        q = q.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)

        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, operations=None): # UNetOps(), ControlnetOps()
        super().__init__()

        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, 
                              context_dim=None, operations=operations)
        self.ff = FeedForward(dim, operations=operations)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, operations=operations)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x, context: Optional[torch.Tensor]):
        n = self.norm1(x)
        n = self.attn1(n, context=None)

        x += n
        n = self.norm2(x)
        n = self.attn2(n, context=context)

        x += n
        x = self.ff(self.norm3(x)) + x
        return x


class TimestepEmbedSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, context_dim=None,
                 operations=None): # UNetOps(), ControlnetOps()

        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = operations.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim[d],
                                   operations=operations)
                for d in range(depth)]
        )
        self.proj_out = operations.Linear(in_channels, inner_dim)

        self.BxCxHxW_BxHWxC = Rearrange('b c h w -> b (h w) c')
        self.BxHxWxC_BxCxHxW = Rearrange('b h w c -> b c h w')

    def forward(self, x, emb, context):
        # xxxx7777: x, [emb], context

        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.BxCxHxW_BxHWxC(x).contiguous()
        x = self.proj_in(x)
            
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        x = self.proj_out(x)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = x.reshape(b, h, w, c)
        x = self.BxHxWxC_BxCxHxW(x).contiguous()
        return x + x_in

