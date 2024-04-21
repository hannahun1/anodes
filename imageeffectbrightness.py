import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectBrightness:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brightness"
    CATEGORY = "ImageEffects_Anodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "brightness_values": ("STRING", {
                    "default": "0:(1.0),\n7:(1.5),\n15:(1.0)\n",
                    "multiline": True
                }),
                "brightness_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01
                }),
            },
        }

    def apply_brightness(self, images, brightness_values, brightness_multiplier):
        points = []
        brightness_values = brightness_values.rstrip(',\n')
        for point_str in brightness_values.split(','):
            frame_str, brightness_str = point_str.split(':')
            frame = int(frame_str.strip())
            brightness = float(brightness_str.strip()[1:-1])
            points.append((frame, brightness))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            brightness_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            brightness_value *= brightness_multiplier
            img = F.adjust_brightness(image, brightness_value)
            out.append(img)

        return (postprocess_frames(torch.stack(out)),)
