# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F

from SDXL.util import (
    DictToClass,
    load_model_weight,
    load_base_clip_model_weight,
    load_refiner_clip_model_weight,
)
from typing import Tuple, Dict

import todos
import pdb


# class QuickGELUActivation(nn.Module):
#     """
#     Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
#     """
#     def forward(self, input):
#         return input * torch.sigmoid(1.702 * input)

# class GELUActivation(nn.Module):
#     def __init__(self, use_gelu_python: bool = False):
#         super().__init__()
#         if use_gelu_python:
#             self.act = self._gelu_python
#         else:
#             self.act = nn.functional.gelu

#     def _gelu_python(self, input):
#         return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

#     def forward(self, input):
#         return self.act(input)


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super(CLIPTextEmbeddings, self).__init__()

        embed_dim = config.hidden_size # 768
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim) # (49408, 768)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim) # (77, 768)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # (self.position_ids -- size(): [1, 77]
        # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        #          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        #          54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        #          72, 73, 74, 75, 76]], device='cuda:0')

    def forward(self, input_tokens):
        inputs_embeds = self.token_embedding(input_tokens)
        position_embeddings = self.position_embedding(self.position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings # size() -- [1, 77, 768]

class CLIPAttention(nn.Module):
    def __init__(self, config):
        super(CLIPAttention, self).__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len: int, B: int):
        return tensor.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, causal_attention_mask):
        """Input shape: Batch x Time x Channel"""
        B, L, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, B)
        value_states = self._shape(self.v_proj(hidden_states), -1, B)

        proj_shape = (B * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, L, B).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # apply the causal_attention_mask first
        attn_weights = attn_weights.view(B, self.num_heads, L, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(B * self.num_heads, L, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=False) # self.training

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(B, self.num_heads, L, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, L, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super(CLIPMLP, self).__init__()
        self.activation_fn = nn.GELU() #  GELUActivation() # QuickGELUActivation()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super(CLIPEncoderLayer, self).__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, causal_attention_mask):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super(CLIPEncoder, self).__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, causal_attention_mask):
        encoder_states = ()
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_output = encoder_layer(hidden_states, causal_attention_mask)
            hidden_states = layer_output

        encoder_states = encoder_states + (hidden_states,)

        return {
            "last_hidden_state" : hidden_states, 
            "hidden_states" : encoder_states,
        }


def make_causal_mask(x):
    """
    Make causal mask used for bi-directional self-attention.
    """
    B, L = x.size() # torch.Size([1, 77])
    # mask = torch.full((L, L), -1000000000.0, device=x.device)
    mask = torch.full((L, L), torch.tensor(torch.finfo(torch.torch.float32).min), device=x.device)

    mask_cond = torch.arange(mask.size(-1), device=x.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(x.dtype) # mask.size() -- [77, 77]

    return mask[None, None, :, :].expand(B, 1, L, L) # size() -- [1, 1, 77, 77]


class CLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super(CLIPTextTransformer, self).__init__()
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tokens):
        input_shape = input_tokens.size() # [1, 77]
        input_tokens = input_tokens.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_tokens) # OK

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = make_causal_mask(input_tokens)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs["last_hidden_state"]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_tokens.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        return {
            "hidden_states": encoder_outputs["hidden_states"],
            "pooler_output": pooled_output,
        }


class CLIPTextModel(nn.Module): 
    def __init__(self, config):
        super(CLIPTextModel, self).__init__()
        self.text_model = CLIPTextTransformer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding # Embedding(49408, 1280)

    def forward(self, input_tokens):
        return self.text_model(input_tokens)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super(CLIPVisionEmbeddings, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

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

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config):
        super(CLIPVisionTransformer, self).__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values) -> Dict[str, torch.Tensor]:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }


class CLIPVisionEncode(nn.Module):
    '''
        CLIPVisionModelWithProjection
    '''
    def __init__(self, config):
        super(CLIPVisionEncode, self).__init__()
        self.vision_model = CLIPVisionTransformer(config)
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        load_model_weight(self, model_path="models/clip_vision_g.safetensors")


    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(self, pixel_values) -> Dict[str, torch.Tensor]:
        # tensor [pixels_values] size: [1, 3, 224, 224] , min: -1.7922625 , max: 2.1458969, mean: -0.467127
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)

        return {
            "image_embeds": image_embeds,
            "last_hidden_state": vision_outputs["last_hidden_state"],
            "hidden_states": vision_outputs["hidden_states"],
            "attentions": vision_outputs["attentions"],
        }


class SDXLClipL(nn.Module):
    '''SD1ClipModel'''

    def __init__(self):
        super(SDXLClipL, self).__init__()
        # comfy/sd1_clip_config.json
        config = DictToClass(
            {
              "_name_or_path": "openai/clip-vit-large-patch14",
              "architectures": [
                "CLIPTextModel"
              ],
              "attention_dropout": 0.0,
              "bos_token_id": 0,
              "dropout": 0.0,
              "eos_token_id": 2,
              "hidden_act": "quick_gelu",
              "hidden_size": 768,
              "initializer_factor": 1.0,
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "layer_norm_eps": 1e-05,
              "max_position_embeddings": 77,
              "model_type": "clip_text_model",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 1,
              "projection_dim": 768,
              "torch_dtype": "float32",
              "transformers_version": "4.24.0",
              "vocab_size": 49408
            }
        )

        self.layer_idx = 11
        self.transformer = CLIPTextModel(config)
        self.text_projection = nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.logit_scale = nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = True

    def forward(self, tokens):
        outputs = self.transformer(tokens)
        z = outputs["hidden_states"][self.layer_idx]
        if self.layer_norm_hidden_state: # True
            z = self.transformer.text_model.final_layer_norm(z)
        pooled = outputs["pooler_output"].float().to(self.text_projection.device) @ self.text_projection.float()

        return z.float(), pooled.float()


class SDXLClipG(nn.Module):
    def __init__(self):
        super(SDXLClipG, self).__init__()
        # clip_config_bigg.json
        config = DictToClass(
            {
              "architectures": [
                "CLIPTextModel"
              ],
              "attention_dropout": 0.0,
              "bos_token_id": 0,
              "dropout": 0.0,
              "eos_token_id": 2,
              "hidden_act": "gelu",
              "hidden_size": 1280,
              "initializer_factor": 1.0,
              "initializer_range": 0.02,
              "intermediate_size": 5120,
              "layer_norm_eps": 1e-05,
              "max_position_embeddings": 77,
              "model_type": "clip_text_model",
              "num_attention_heads": 20,
              "num_hidden_layers": 32,
              "pad_token_id": 1,
              "projection_dim": 1280,
              "torch_dtype": "float32",
              "vocab_size": 49408
            }
        )
        self.layer_idx = -2
        self.transformer = CLIPTextModel(config)
        self.text_projection = nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.logit_scale = nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = False

    def forward(self, tokens):
        # tensor([49406,  3365,   267,  3772,  5994,   267,  1105,   633, 14559, 49407,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0], device='cuda:0')
        outputs = self.transformer(tokens)
        z = outputs["hidden_states"][self.layer_idx]
        if self.layer_norm_hidden_state: # False
            z = self.transformer.text_model.final_layer_norm(z)
        pooled = outputs["pooler_output"].float().to(self.text_projection.device) @ self.text_projection.float()

        return z.float(), pooled.float()


class CLIPTextEncode(nn.Module):
    '''CLIPTextEncode'''

    def __init__(self, version="base_1.0"):
        super(CLIPTextEncode, self).__init__()
        self.version = version

        if version == "base_1.0":
            self.clip_l = SDXLClipL()
            self.clip_g = SDXLClipG()
        else: # refiner_1.0
            self.clip_g = SDXLClipG()

        if version == "base_1.0":
            load_base_clip_model_weight(self, model_path="models/sd_xl_base_1.0.safetensors")
        else:
            load_refiner_clip_model_weight(self, model_path="models/sd_xl_refiner_1.0.safetensors")

    def forward(self, tokens, device=torch.device('cuda')):
        if self.version == "base_1.0":
            token_l = tokens['l']  # padding with stop_token
            token_g = tokens['g']  # padding with 0

            l_out, l_pooled = self.clip_l(torch.LongTensor(token_l).to(device))
            g_out, g_pooled = self.clip_g(torch.LongTensor(token_g).to(device))

            return torch.cat([l_out, g_out], dim=-1), g_pooled

        # refiner_1.0 version
        token_g = tokens['g']
        g_out, g_pooled = self.clip_g(torch.LongTensor(token_g).to(device))
        return g_out, g_pooled


def clip_vision_model():
    # output: SdxlClipVision

    '''This model is been used by sdxl 1.0 base model'''

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
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 48,
            "patch_size": 14,
            "projection_dim": 1280,
            "torch_dtype": "float32",
        }
    )
    model = CLIPVisionEncode(config)
    model = model.half()
    model = model.eval()
    return model   

def clip_text_model(version="base_1.0"):
    # output: SdxlClipText
    # output: RefinerClipText

    model = CLIPTextEncode(version=version)
    model = model.half()
    model = model.eval()
    return model  


if __name__ == "__main__":
    import todos

    # model = clip_vision_model()

    # model = clip_text_model(version="base_1.0")
    model = clip_text_model(version="refiner_1.0")

    # model = torch.jit.script(model)
    # print(model)
    todos.debug.output_weight(model.state_dict())

