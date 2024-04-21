import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectHue:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_hue"
    CATEGORY = "ImageEffects_Anodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "hue_values": ("STRING", {
                    "default": "0:(0.0),\n7:(0.5),\n15:(1.0)\n",
                    "multiline": True
                }),
                "hue_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01
                }),
            },
        }

    def apply_hue(self, images, hue_values, hue_multiplier):
        points = []
        hue_values = hue_values.rstrip(',\n')
        for point_str in hue_values.split(','):
            frame_str, hue_str = point_str.split(':')
            frame = int(frame_str.strip())
            # Scale input to range -0.5 to 0.5
            hue = (float(hue_str.strip()[1:-1]) - 0.5) * 2 * 0.5
            points.append((frame, hue))

        points.sort(key=lambda x: x[0])  # Make sure points are in order
        images = preprocess_frames(images)
        out = []

        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            hue_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            # Apply the multiplier and ensure the value is within the -0.5 to 0.5 range
            hue_value = (hue_value * hue_multiplier) * 0.5
            hue_value = max(-0.5, min(0.5, hue_value))

            img = image.clone()  # Clone to avoid modifying the original in-place
            img = F.adjust_hue(img, hue_value)  # Apply hue adjustment

            out.append(img)

        return (postprocess_frames(torch.stack(out)),)
