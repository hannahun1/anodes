import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectRed:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_red"
    CATEGORY = "ImageEffects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "red_values": ("STRING", {
                    "default": "0:(1.0),\n7:(0.5),\n15:(1.0)\n",
                    "multiline": True 
                }),
                "red_multiplier": ("FLOAT", {
                    "default": 1.0,  # Default multiplier value
                    "step": 0.01
                }),
            },
        }

    def apply_red(self, images, red_values, red_multiplier):
        points = []
        red_values = red_values.rstrip(',\n')
        for point_str in red_values.split(','):
            frame_str, red_str = point_str.split(':')
            frame = int(frame_str.strip())
            red = float(red_str.strip()[1:-1])
            points.append((frame, red))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            red_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            red_value *= red_multiplier  # Multiply the red value by the multiplier

            # Adjust the red channel specifically
            img = image.clone()  # Clone to avoid modifying the original in-place
            r = img[0] * red_value
            g = img[1]  # Green channel remains the same
            b = img[2]  # Blue channel remains the same

            # Stack channels back together
            modified_img = torch.stack([r, g, b], dim=0)
            out.append(modified_img)

        return (postprocess_frames(torch.stack(out)),)