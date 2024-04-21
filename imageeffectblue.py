import torch
import torchvision.transforms.functional as F
import numpy as np
import einops


def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectBlue:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_blue"
    CATEGORY = "ImageEffects_Anodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blue_values": ("STRING", {
                    "default": "0:(1.0),\n7:(0.5),\n15:(1.0)\n",
                    "multiline": True
                }),
                "blue_multiplier": ("FLOAT", {
                    "default": 1.0,  # Default multiplier value
                    "step": 0.01
                }),
            },
        }

    def apply_blue(self, images, blue_values, blue_multiplier):
        points = []
        blue_values = blue_values.rstrip(',\n')
        for point_str in blue_values.split(','):
            frame_str, blue_str = point_str.split(':')
            frame = int(frame_str.strip())
            blue = (float(blue_str.strip()[1:-1]) * 2 - 1) * 0.5  # Rescale to [-0.5, 0.5]
            points.append((frame, blue))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            blue_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            blue_value *= blue_multiplier

            img = image.clone()  # Clone to avoid modifying the original in-place
            r, g, b = img[0], img[1], img[2] * blue_value  # Apply blue multiplier
            modified_img = torch.stack([r, g, b], dim=0)

            out.append(modified_img)

        return (postprocess_frames(torch.stack(out)),)