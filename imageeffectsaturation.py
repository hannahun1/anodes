import torch
import torchvision.transforms.functional as F
import numpy as np
import einops

def preprocess_frames(frames):
    return einops.rearrange(frames, "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

class ImageEffectSaturation:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_saturation"
    CATEGORY = "ImageEffects_Anodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "saturation_values": ("STRING", {
                    "default": "0:(1.0),\n7:(1.5),\n15:(1.0)\n",
                    "multiline": True
                }),
                "saturation_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01
                }),
            },
        }

    def apply_saturation(self, images, saturation_values, saturation_multiplier):
        points = []
        saturation_values = saturation_values.rstrip(',\n')
        for point_str in saturation_values.split(','):
            frame_str, saturation_str = point_str.split(':')
            frame = int(frame_str.strip())
            saturation = float(saturation_str.strip()[1:-1])
            points.append((frame, saturation))

        points.sort(key=lambda x: x[0])

        images = preprocess_frames(images)
        out = []
        for idx, image in enumerate(images):
            current_frame_index = idx % len(points)
            saturation_value = np.interp(current_frame_index, [p[0] for p in points], [p[1] for p in points])
            saturation_value *= saturation_multiplier
            img = F.adjust_saturation(image, saturation_value)
            out.append(img)

        return (postprocess_frames(torch.stack(out)),)
