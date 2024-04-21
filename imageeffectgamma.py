import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectGamma:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_gamma"
    CATEGORY = "ImageEffects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "gamma_values": ("STRING", {
                    "default": "0:(1.0),\n7:(0.8),\n15:(2.2)\n",
                    "multiline": True
                }),
                "gamma_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    def apply_gamma(self, images, gamma_values, gamma_multiplier):
        points = []
        gamma_values = gamma_values.rstrip(',\n')
        for point_str in gamma_values.split(','):
            frame_str, gamma_str = point_str.split(':')
            frame = int(frame_str.strip())
            gamma = float(gamma_str.strip()[1:-1])
            points.append((frame, gamma))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            gamma_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            gamma_value *= gamma_multiplier
            img = F.adjust_gamma(image, gamma_value)
            out.append(img)

        return (postprocess_frames(torch.stack(out)),)
