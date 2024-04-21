import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectSharpness:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sharpness"
    CATEGORY = "ImageEffects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sharpness_values": ("STRING", {
                    "default": "0:(1.0),\n7:(2.0),\n15:(0.5)\n",
                    "multiline": True
                }),
                "sharpness_multiplier": ("FLOAT", {
                    "default": 1.0,  # Default multiplier value
                    "min": 0.0,      # Minimum sharpness multiplier
                    "max": 3.0,      # Maximum sharpness multiplier
                    "step": 0.1
                }),
            },
        }

    def apply_sharpness(self, images, sharpness_values, sharpness_multiplier):
        points = []
        sharpness_values = sharpness_values.rstrip(',\n')
        for point_str in sharpness_values.split(','):
            frame_str, sharpness_str = point_str.split(':')
            frame = int(frame_str.strip())
            sharpness = float(sharpness_str.strip()[1:-1])
            points.append((frame, sharpness))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            sharpness_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            sharpness_value *= sharpness_multiplier
            img = F.adjust_sharpness(image, sharpness_value)
            out.append(img)

        return (postprocess_frames(torch.stack(out)),)