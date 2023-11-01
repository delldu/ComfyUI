# coding=utf-8
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

# import SDXL.util

from SDXL.util import (
    DictToClass,
    load_model_weight,
    load_base_clip_model_weight,
    load_refiner_clip_model_weight,
    # Linear,
    # Conv2d,
    load_clip_vision_image,
)
from SDXL.tokenizer import (
    CLIPTextTokenizer,
)

from typing import Tuple, Dict

import todos
import pdb

class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        # use_gelu_python = True
        # print("------------ use_gelu_python ", use_gelu_python)
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)

    def __repr__(self):
        return f"GELUActivation({self.act})"

class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)



class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

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
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # self.k_proj = SDXL.util.Linear(self.embed_dim, self.embed_dim)
        # self.v_proj = SDXL.util.Linear(self.embed_dim, self.embed_dim)
        # self.q_proj = SDXL.util.Linear(self.embed_dim, self.embed_dim)
        # self.out_proj = SDXL.util.Linear(self.embed_dim, self.embed_dim)

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
        if causal_attention_mask is not None:
            attn_weights = attn_weights.view(B, self.num_heads, L, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(B * self.num_heads, L, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # self.training

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(B, self.num_heads, L, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, L, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act == "quick_gelu": # SDXLClipL
            self.activation_fn = QuickGELUActivation()
        else:
            self.activation_fn = GELUActivation() # nn.GELU() #  QuickGELUActivation()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        # self.fc1 = SDXL.util.Linear(config.hidden_size, config.intermediate_size)
        # self.fc2 = SDXL.util.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, causal_attention_mask=None):
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
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, causal_attention_mask=None):
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
        super().__init__()
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

        pool_encoded = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_tokens.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        return {
            "hidden_states": encoder_outputs["hidden_states"],
            "pool_encoded": pool_encoded,
        }


class CLIPTextModel(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.text_model = CLIPTextTransformer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding # Embedding(49408, 1280)

    def forward(self, input_tokens):
        return self.text_model(input_tokens)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
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

        # self.patch_embedding = SDXL.util.Conv2d(
        #     in_channels=config.num_channels,
        #     out_channels=self.embed_dim,
        #     kernel_size=self.patch_size,
        #     stride=self.patch_size,
        #     bias=False,
        # )

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
        super().__init__()
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

        last_hidden_state = encoder_outputs["last_hidden_state"]
        pool_encoded = last_hidden_state[:, 0, :]
        pool_encoded = self.post_layernorm(pool_encoded)

        return {
            "last_hidden_state": last_hidden_state,
            "pool_encoded": pool_encoded,
            "hidden_states": encoder_outputs["hidden_states"],
        }


class CLIPVisionEncode(nn.Module):
    '''
        CLIPVisionModelWithProjection
    '''
    def __init__(self):
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
                "model_type": "clip_vision_model",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 48,
                "patch_size": 14,
                "projection_dim": 1280,
                "torch_dtype": "float32",
            }
        )
        self.vision_model = CLIPVisionTransformer(config)
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        # self.visual_projection = SDXL.util.Linear(config.hidden_size, config.projection_dim, bias=False)

        self.normal = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        for param in self.parameters():
            param.requires_grad = False

        load_model_weight(self, model_path="models/clip_vision_g.safetensors")


    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(self, pixel_values, normal_input=True): # -> Dict[str, torch.Tensor]:
        if normal_input:
            pixel_values = F.interpolate(pixel_values, size=(224, 224), mode="bilinear", align_corners=False)
            # image normal
            pixel_values = self.normal(pixel_values)
            # tensor [pixel_values] size: [1, 3, 224, 224], min: -1.792263, max: 2.145897, mean: -0.467128

        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pool_encoded = vision_outputs["pool_encoded"]  # pool_encoded
        image_embeds = self.visual_projection(pool_encoded)

        return image_embeds.to(torch.float32)

class SDXLClipL(nn.Module):
    '''SD1ClipModel'''

    def __init__(self):
        super().__init__()
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
        self.layer_norm_hidden_state = False # True

    def forward(self, tokens):
        outputs = self.transformer(tokens)
        z = outputs["hidden_states"][self.layer_idx]
        if self.layer_norm_hidden_state: # False
            z = self.transformer.text_model.final_layer_norm(z)

        # tensor [SD1ClipModel z] size: [2, 77, 768], min: -809.318359, max: 853.722839, mean: 0.013015
        # todos.debug.output_var("SDXLClipL z", z) # xxxx9999
        # pdb.set_trace()


        pooled = outputs["pool_encoded"].float().to(self.text_projection.device) @ self.text_projection.float()

        return z.float(), pooled.float()


class SDXLClipG(nn.Module):
    def __init__(self):
        super().__init__()
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

        # todos.debug.output_var("SDXLClipG z", z) # xxxx9999
        # # OK tensor [SD1ClipModel z] size: [2, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.027165

        # pdb.set_trace()

        pooled = outputs["pool_encoded"].float().to(self.text_projection.device) @ self.text_projection.float()

        return z.float(), pooled.float()


class CLIPTextEncode(nn.Module):
    '''CLIPTextEncode'''

    def __init__(self, version):
        super().__init__()
        self.version = version

        if version == "base_1.0":
            self.clip_l = SDXLClipL()
            self.clip_g = SDXLClipG()
        else: # refiner_1.0
            self.clip_g = SDXLClipG()

        for param in self.parameters():
            param.requires_grad = False

        # if version == "base_1.0":
        #     load_base_clip_model_weight(self, model_path="models/sd_xl_base_1.0.safetensors")
        # else:
        #     load_refiner_clip_model_weight(self, model_path="models/sd_xl_refiner_1.0.safetensors")

    def forward(self, tokens, device=torch.device('cpu')):
        if self.version == "base_1.0":
            token_l = tokens['l']  # padding with stop_token
            token_g = tokens['g']  # padding with 0

            l_out, l_pooled = self.clip_l(torch.LongTensor(token_l).to(device))
            g_out, g_pooled = self.clip_g(torch.LongTensor(token_g).to(device))

            # return torch.cat([l_out, g_out], dim=-1), g_pooled
            return {
                "text_encoded" : torch.cat([l_out, g_out], dim=-1), 
                "pool_encoded" : g_pooled
            }

        # refiner_1.0 version
        token_g = tokens['g']
        g_out, g_pooled = self.clip_g(torch.LongTensor(token_g).to(device))

        return {
            "text_encoded" : g_out, 
            "pool_encoded" : g_pooled
        }

def create_clip_text_model(version):
    # output: SdxlClipText
    # output: RefinerClipText

    model = CLIPTextEncode(version=version)
    # model = model.half()
    model = model.eval()
    # model = model.cuda()
    return model  

def create_clip_token_model(version):
    return CLIPTextTokenizer(version=version)


def test_old_clip_vision():
    from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig

    config = CLIPVisionConfig.from_json_file("../../comfy/clip_vision_config_g.json")
    model = CLIPVisionModelWithProjection(config)
    load_model_weight(model, model_path="models/clip_vision_g.safetensors")
    model.half().eval().cuda()

    return model

def clip_vision_model():
    # output: SdxlClipVision

    '''This model is been used by sdxl 1.0 base model'''

    model = CLIPVisionEncode()
    model.half().eval().cuda()
    return model

if __name__ == "__main__":
    import todos

    # # model_version = "base_1.0"
    # model_version = "refiner_1.0"
    # print(f"model version {model_version}")

    # model = create_clip_text_model(version=model_version)
    # # model = model.cuda()

    # # model = torch.jit.script(model)
    # # print(model)
    # # todos.debug.output_weight(model.state_dict())

    # positive_prompt = "bag, clean background, made from cloth"
    # negative_prompt = "watermark, text"

    # clip_token = CLIPTextTokenizer(version=model_version)
    # positive_tokens = clip_token.encode(positive_prompt)
    # print(positive_tokens)

    # with torch.no_grad():
    #     positive_tensor = model(positive_tokens)
    # todos.debug.output_var("positive_tensor", positive_tensor)
    # print(
    #     '''
    #     # Standard
    #     # tensor [cond] size: [1, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.035082, positive_output_tensor
    #     # tensor [pooled] size: [1, 1280], min: -3.958434, max: 3.203121, mean: 0.009559
    #     '''
    # )

    # negative_tokens = clip_token.encode(negative_prompt)
    # print(negative_tokens)
    # with torch.no_grad():
    #     negative_tensor = model(negative_tokens)
    # todos.debug.output_var("negative_tensor", negative_tensor)

    # print(
    #     '''
    #     # Standard 
    #     # tensor [cond] size: [1, 77, 1280], min: -66.179367, max: 18.368397, mean: 0.029526, negative_output_tensor
    #     # tensor [pooled] size: [1, 1280], min: -3.58707, max: 3.409507, mean: 0.024002
    #     '''
    # )

    model = test_old_clip_vision()

    # todos.debug.output_weight(model.state_dict())
    image = load_clip_vision_image("../workflow/image.png")
    todos.debug.output_var("image", image)

    print("Old implement ...")
    with torch.no_grad():
        output = model(image.half().cuda(), output_hidden_states=True)
        # output = model(image, output_hidden_states=True)
    todos.debug.output_var("output", output)

    print("-" * 120)

    print("New implement ...")
    model = clip_vision_model()
    with torch.no_grad():
        output = model(image.half().cuda(), normal_input=False)
    todos.debug.output_var("output/image_embeds", output)

    # tensor [pixel_values] size: [1, 3, 224, 224], min: -1.791992, max: 2.146484, mean: -0.467529
    # tensor [pool_encoded] size: [1, 1664], min: -8.867188, max: 7.449219, mean: 0.144165
    # tensor [image_embeds] size: [1, 1280], min: -6.214844, max: 4.339844, mean: -0.038574
