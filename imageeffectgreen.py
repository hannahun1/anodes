import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

# Preprocess and Postprocess functions as defined
def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectGreen:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_green"
    CATEGORY = "ImageEffects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "green_values": ("STRING", {
                    "default": "0:(1.0),\n7:(0.5),\n15:(1.0)\n",
                    "multiline": True
                }),
                "green_multiplier": ("FLOAT", {
                    "default": 1.0,  # Default multiplier value
                    "step": 0.01
                }),
            },
        }

    def apply_green(self, images, green_values, green_multiplier):
        points = []
        green_values = green_values.rstrip(',\n')
        for point_str in green_values.split(','):
            frame_str, green_str = point_str.split(':')
            frame = int(frame_str.strip())
            green = float(green_str.strip()[1:-1])
            points.append((frame, green))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            green_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            green_value *= green_multiplier  # Apply the green multiplier

            # Adjust the green channel specifically
            img = image.clone()  # Clone to avoid modifying the original in-place
            r = img[0]  # Red channel remains the same
            g = img[1] * green_value  # Apply green multiplier
            b = img[2]  # Blue channel remains the same

            # Stack channels back together
            modified_img = torch.stack([r, g, b], dim=0)
            out.append(modified_img)

        return (postprocess_frames(torch.stack(out)),)