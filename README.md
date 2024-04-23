# Image Effect Scheduler Node Set for ComfyUI

![image](https://github.com/hannahunter88/anodes/blob/master/2024-04-21%20092318.png)

ImageEffect nodes were created to provide complete control over visual effects by allowing users to specify individual values for each frame. For example it can be useful in creating visual effects that are synchronized with extracted music features.

ImageEffect nodes were also created to enable the creation of smoother, more consistent visual effects by automatically interpolating values between frames when users provide fewer specific values.

*Installation*

Clone this repo into custom_nodes folder.

*How to use these nodes?*

ImageEffect nodes should be applied after KSampler and VAE decode and before Preview Images and Video Combine nodes. 
It is also recommended to experiment with multiple ImageEffect nodes simultaneously, while keeping the original preview images and video for comparison.

Values in the list should be specified as floating-point numbers between 0 and 1. Use the multiplier parameter if you need higher values. For ImageEffectHue, the allowed range is -0.5 to 0.5, but these values are also rescaled to fit with the other ImageEffect Nodes, so you can still work with floating-point values between 0 and 1.

**audiofeatures_anodes.ipynb**

If you're feeling adventurous, you can play around with audiofeatures_anodes.ipynb. In addition to the standard beat times lists, I have added a new set of lists generated using librosa for advanced audio feature extraction in the notebook. These lists include Mel-Frequency Cepstral Coefficients (MFCC), their deltas, and Chroma features, all resampled to a frame rate of 10 FPS to seamlessly integrate with the existing setup. You can use the values from these lists with the ImageEffect nodes after KSampler (and VAE Decode) or with Kijai's CreateFadeMaskAdvanced node before KSampler, for example, to manipulate the IPAdapter.

**The video combine node requires a frame rate of 10 FPS when using these lists!**

I just shared this as an example, but there are many other ways to create lists. I encourage you to experiment and explore!

[![ImageEffectHue Node with Beattimes_Switch List](https://img.youtube.com/vi/xSIVtJ7xsHY/maxresdefault.jpg)](https://www.youtube.com/watch?v=xSIVtJ7xsHY)
