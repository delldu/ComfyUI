# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#
import random
import gradio as gr
import numpy as np

import torch
import einops

import cv2
import SDXL
# from SDXL.util import (
# )

import pdb


def ImageResize(image):
    MAX_HW = 1024
    H, W, C = image.shape

    if H > MAX_HW or W > MAX_HW:
        s = min(MAX_H / H, MAX_W / W)
        H, W = int(s * H), int(s * W)

    NH, NW = (H // 8) * 8, (W // 8) * 8
    if NH != H or NW != W:
        return cv2.resize(image, (NW, NH))

    return image

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


guide_generator = CannyDetector()
# model, device = SDXL.get_model('v1.0')

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, time_steps, strength, 
    scale, seed, low_threshold, high_threshold):

    if input_image is not None:
        input_image = ImageResize(input_image)

    if seed == -1:
        seed = random.randint(0, 2147483647)    

    if input_image is not None:
        guide_image = HWC3(guide_generator(input_image, low_threshold, high_threshold))
    else:
        guide_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)

    return [255 - guide_image]



    with torch.no_grad():
        if input_image is not None:
            input_image = cv2.resize(input_image, (W, H))
            canny_detected_map = HWC3(guide_generator(HWC3(input_image), low_threshold, high_threshold))
        else:
            canny_detected_map = np.zeros((H, W, C)).astype(np.uint8)

        mlsd_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        hed_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        sketch_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        openpose_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        midas_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        seg_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        content_emb = np.zeros((768))

        detected_maps_list = [canny_detected_map, 
                              mlsd_detected_map, 
                              hed_detected_map,
                              sketch_detected_map,
                              openpose_detected_map,
                              midas_detected_map,
                              seg_detected_map                          
                              ]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()

        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        # ['a diagram, best quality, extremely detailed']
        p_tokens = CLIP.tokenize([prompt + ', ' + a_prompt] * num_samples).cuda()
        n_tokens = CLIP.tokenize([n_prompt] * num_samples).cuda()

        # (Pdb) local_control.size(), local_control.dtype, local_control.min(), local_control.max()
        # ([1, 21, 640, 512], torch.float32, 0., 1.)

        with torch.no_grad():        
            x_samples = model(local_control, global_control, p_tokens, n_tokens, time_steps,
                strength = strength, c_guide_scale = scale, seed = seed)

        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]

    return [255 - guide_image] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## SDXL Base Model (Version 1.0) Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy", label='canny')

            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1024, value=1024, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                time_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=8.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=42, step=1)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", 
                columns=[2], rows=[2], object_fit="contain", height="auto")

    inputs = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, time_steps, 
              strength, scale, seed, low_threshold, high_threshold]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')
