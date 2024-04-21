from .imageeffectcontrast import ImageEffectContrast
from .imageeffectbrightness import ImageEffectBrightness
from .imageeffectsaturation import ImageEffectSaturation
from .imageeffecthue import ImageEffectHue
from .imageeffectgamma import ImageEffectGamma
from .imageeffectsharpness import ImageEffectSharpness
from .imageeffectred import ImageEffectRed
from .imageeffectgreen import ImageEffectGreen
from .imageeffectblue import ImageEffectBlue

NODE_CLASS_MAPPINGS = {
    "ImageEffectContrast": ImageEffectContrast,
    "ImageEffectBrightness": ImageEffectBrightness,
    "ImageEffectSaturation": ImageEffectSaturation,
    "ImageEffectHue": ImageEffectHue,
    "ImageEffectGamma": ImageEffectGamma,
    "ImageEffectSharpness": ImageEffectSharpness,
    "ImageEffectRed": ImageEffectRed,
    "ImageEffectGreen": ImageEffectGreen,
    "ImageEffectBlue": ImageEffectBlue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageEffectContrast": "ImageEffectContrast",
    "ImageEffectBrightness": "ImageEffectBrightness",
    "ImageEffectSaturation": "ImageEffectSaturation",
    "ImageEffectHue": "ImageEffectHue",
    "ImageEffectGamma": "ImageEffectGamma",
    "ImageEffectSharpness": "ImageEffectSharpness",
    "ImageEffectRed": "ImageEffectRed",
    "ImageEffectGreen": "ImageEffectGreen",
    "ImageEffectBlue": "ImageEffectBlue"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']