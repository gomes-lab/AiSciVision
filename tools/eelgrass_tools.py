"""
Holds all available tools for the eelgrass wasting disease classification task.
"""

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import config
from models.clip_classifier import CLIPMLPImageClassifier

from . import Tool


class AdjustBrightnessTool(Tool):
    description = "Tool Name: 'AdjustBrightnessTool' Description: Adjusts the brightness of the image by 50%. This tool can help when the image is too dark or too bright, allowing for better visibility of disease symptoms on the eelgrass blade."

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        enhancer = ImageEnhance.Brightness(image)
        enhanced_image = enhancer.enhance(1.5)  # Increase brightness by 50%
        return "Image brightness has been increased by 50%.", enhanced_image, {}


class SharpenTool(Tool):
    description = "Tool Name: 'SharpenTool' Description: Sharpens the image to enhance edges and details. This tool is useful for making subtle features more prominent, which can help in identifying signs of eelgrass wasting disease."

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        enhancer = ImageEnhance.Sharpness(image)
        enhanced_image = enhancer.enhance(2.0)  # Increase sharpness level
        return "Image sharpness has been increased.", enhanced_image, {}


class EdgeDetectionTool(Tool):
    description = "Tool Name: 'EdgeDetectionTool' Description: Applies edge detection to the image, highlighting boundaries and features. This can help in identifying lesions or patterns associated with eelgrass wasting disease."

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        edge_image = image.convert("L").filter(ImageFilter.FIND_EDGES)
        return "Edge detection has been applied to the image.", edge_image, {}


class IncreaseContrastTool(Tool):
    description = "Tool Name: 'IncreaseContrastTool' Description: Increases the contrast of the image by 50%. This tool can be helpful when the image appears too flat or when you need to enhance the visibility of subtle details, especially in cases where eelgrass wasting disease symptoms might be hard to distinguish."

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(1.5)  # Increase contrast by 50%
        return "Image contrast has been increased by 50%.", enhanced_image, {}


class DecreaseContrastTool(Tool):
    description = "Tool Name: 'DecreaseContrastTool' Description: Decreases the contrast of the image by 50%. This tool can be useful when the image appears too harsh or when you want to reduce the intensity of bright areas, which might help in identifying overall patterns or structures in the eelgrass."

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(0.5)  # Decrease contrast by 50%
        return "Image contrast has been decreased by 50%.", enhanced_image, {}


class HistogramEqualizationTool(Tool):
    description = "Tool Name: 'HistogramEqualizationTool' Description: Enhances the contrast of the image using histogram equalization. This can help in making features more distinguishable, which is beneficial for detecting eelgrass wasting disease symptoms."

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        image_gray = image.convert("L")
        hist_equalized = ImageOps.equalize(image_gray)
        return "Histogram equalization has been applied to enhance image contrast.", hist_equalized, None


class PredictEelgrassWastingDiseaseTool(Tool):
    description = "Tool Name: 'PredictEelgrassWastingDiseaseTool' Description: Predicts the probability of eelgrass wasting disease in the image using a machine learning model. I reccomend you use this tool."

    def __init__(self, clip_path: str):
        clf = CLIPMLPImageClassifier(clip_model_config=config.clip_model_config, num_labels=2)
        clf.load_model_inplace(clip_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = clf

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        probability = self.model.predict_proba(image)
        return (
            f"The probability of eelgrass wasting disease in this image is {probability:.2f}%. (Percentage out of 100%) Please note that this model has been trained on a limited dataset and its predictions can be inaccurate. Use this information as a guideline rather than a definitive diagnosis.",
            None,
            {},
        )


tool_name2Tool_cls: dict[str, Tool] = {
    "IncreaseContrastTool": IncreaseContrastTool,
    "DecreaseContrastTool": DecreaseContrastTool,
    "PredictEelgrassWastingDiseaseTool": PredictEelgrassWastingDiseaseTool,
    "AdjustBrightnessTool": AdjustBrightnessTool,
    "SharpenTool": SharpenTool,
    "EdgeDetectionTool": EdgeDetectionTool,
    "HistogramEqualizationTool": HistogramEqualizationTool,
}
