import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectContrast:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_contrast"
    CATEGORY = "ImageEffects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "contrast_values": ("STRING", {
                    "default": "0:(1.0),\n7:(0.5),\n15:(1.0)\n",
                    "multiline": True
                }),
                "contrast_multiplier": ("FLOAT", {
                    "default": 1.0,  # Default multiplier value
                    "step": 0.01
                }),
            },
        }

    def apply_contrast(self, images, contrast_values, contrast_multiplier):
        points = []
        contrast_values = contrast_values.rstrip(',\n')
        for point_str in contrast_values.split(','):
            frame_str, contrast_str = point_str.split(':')
            frame = int(frame_str.strip())
            contrast = float(contrast_str.strip()[1:-1])
            points.append((frame, contrast))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            contrast_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            contrast_value *= contrast_multiplier  # Apply the contrast multiplier
            img = F.adjust_contrast(image, contrast_value)
            out.append(img)

        return (postprocess_frames(torch.stack(out)),)
