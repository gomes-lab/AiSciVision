"""
Holds all available tools for the aquaculture pond detection task.
"""

import io
import os

import requests
import torch
from PIL import Image

import config
from models.clip_classifier import CLIPMLPImageClassifier

from . import Tool


def get_google_maps_image(lat: float, lon: float, zoom: int = 20, size: tuple[int, int] = (640, 640)) -> Image.Image:
    api_key = os.getenv("GMAPS_API_KEY")
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&key={api_key}"
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))


def adjust_coordinates(lat: float, lon: float, direction: str, amount: float) -> tuple[float, float]:
    """
    Returns new coordinates (lat, long) by moving latitude and longitude in appropriate direction by some amount.
    Direction either of "up", "down", "left", "right".
    """
    if direction == "left":
        return lat, lon - amount
    elif direction == "right":
        return lat, lon + amount
    elif direction == "up":
        return lat + amount, lon
    elif direction == "down":
        return lat - amount, lon
    else:
        raise ValueError(f"Direction {direction} not supported.")


class PredictAquaculturePondTool(Tool):
    description = "Tool Name: 'PredictAquaculturePondTool' Description: Predicts the probability of an aquaculture pond being present in the image using a machine learning model. This tool is particularly helpful when you need a quantitative assessment of the likelihood of aquaculture pond presence in the satellite image. The model has been trained on various satellite images of aquaculture ponds and provides a percentage probability. However, please note that the model can be wrong or inaccurate, especially in complex or ambiguous cases. It's important to use this tool's output as one piece of evidence among others, and not rely on it exclusively for your final decision."

    def __init__(self, clip_path: str):
        clf = CLIPMLPImageClassifier(clip_model_config=config.clip_model_config, num_labels=2)
        clf.load_model_inplace(clip_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = clf

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        probability = self.model.predict_proba(image)
        return (
            f"The probability of an aquaculture pond being present in this image is {probability:.2f}%. (Percentage out of 100%) Please note that this model has been trained on a limited dataset and its predictions can be inaccurate. Use this information as a guideline rather than a definitive prediction.",
            None,
            md,
        )


class BasePanTool(Tool):
    def __init__(self, direction: str, relative: bool):
        """
        Args:
            direction (str): Direction for panning, either of "up", "down", "left", "right".
            relative (bool): Flag to pan relative to current image, or to original image (absolute).
        """
        assert direction in ("up", "down", "left", "right")

        self.direction = direction
        self.relative = relative

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        if self.relative:
            lat, lon = md["current_lat"], md["current_lon"]
        else:
            lat, lon = md["original_lat"], md["original_lon"]

        new_lat, new_lon = adjust_coordinates(lat, lon, self.direction, 0.001)
        new_image = get_google_maps_image(new_lat, new_lon, md["zoom"])

        md["current_lat"], md["current_lon"] = new_lat, new_lon

        return (
            (
                f"The view has been panned {self.direction}, revealing more of the area {self.direction} of the "
                f"{'previous' if self.relative else 'original'} view. "
                f"New coordinates: Lat {new_lat}, Lon {new_lon}"
            ),
            new_image,
            md,
        )


class PanUpToolRelative(BasePanTool):
    description = "Pans the view upwards relative to the last image seen."

    def __init__(self):
        super().__init__("up", True)


class PanUpToolAbsolute(BasePanTool):
    description = "Pans the view upwards relative to the original starting image."

    def __init__(self):
        super().__init__("up", False)


class PanDownToolRelative(BasePanTool):
    description = "Pans the view downwards relative to the last image seen."

    def __init__(self):
        super().__init__("down", True)


class PanDownToolAbsolute(BasePanTool):
    description = "Pans the view downwards relative to the original starting image."

    def __init__(self):
        super().__init__("down", False)


class PanLeftToolRelative(BasePanTool):
    description = "Pans the view to the left relative to the last image seen."

    def __init__(self):
        super().__init__("left", True)


class PanLeftToolAbsolute(BasePanTool):
    description = "Pans the view to the left relative to the original starting image."

    def __init__(self):
        super().__init__("left", False)


class PanRightToolRelative(BasePanTool):
    description = "Pans the view to the right relative to the last image seen."

    def __init__(self):
        super().__init__("right", True)


class PanRightToolAbsolute(BasePanTool):
    description = "Pans the view to the right relative to the original starting image."

    def __init__(self):
        super().__init__("right", False)


class BaseZoomTool(Tool):
    def __init__(self, zoom_direction: str, relative: bool):
        """
        Args:
            zoom_direction (str): Either "in" or "out".
            relative (bool): Flag to zoom relative to current image, or to original image (absolute).
        """
        assert zoom_direction in ("in", "out")

        self.zoom_direction = zoom_direction
        self.relative = relative

    def __call__(self, image: Image.Image, md: dict) -> tuple[str, Image.Image | None, dict]:
        if self.relative:
            lat, lon = md["current_lat"], md["current_lon"]
            zoom = md["zoom"]
        else:
            lat, lon = md["original_lat"], md["original_lon"]
            zoom = md["original_zoom"]

        new_zoom = zoom + (1 if self.zoom_direction == "in" else -1)
        new_zoom = max(0, min(new_zoom, 21))  # Ensure zoom level is between 0 and 21

        new_image = get_google_maps_image(lat, lon, new_zoom)

        md["zoom"] = new_zoom

        return (
            (
                f"The view has been zoomed {self.zoom_direction}, providing a "
                f"{'closer' if self.zoom_direction == 'in' else 'wider'} look at the "
                f"{'current' if self.relative else 'original'} view. "
                f"New zoom level: {new_zoom}"
            ),
            new_image,
            md,
        )


class ZoomInToolRelative(BaseZoomTool):
    description = "Zooms in on the center of the current view relative to the last image seen."

    def __init__(self):
        super().__init__("in", True)


class ZoomInToolAbsolute(BaseZoomTool):
    description = "Zooms in on the center of the original view relative to the original starting image."

    def __init__(self):
        super().__init__("in", False)


class ZoomOutToolRelative(BaseZoomTool):
    description = "Zooms out from the current view relative to the last image seen."

    def __init__(self):
        super().__init__("out", True)


class ZoomOutToolAbsolute(BaseZoomTool):
    description = "Zooms out from the original view relative to the original starting image."

    def __init__(self):
        super().__init__("out", False)


tool_name2Tool_cls: dict[str, Tool] = {
    "PredictAquaculturePondTool": PredictAquaculturePondTool,
    "PanUpToolRelative": PanUpToolRelative,
    "PanUpToolAbsolute": PanUpToolAbsolute,
    "PanDownToolRelative": PanDownToolRelative,
    "PanDownToolAbsolute": PanDownToolAbsolute,
    "PanLeftToolRelative": PanLeftToolRelative,
    "PanLeftToolAbsolute": PanLeftToolAbsolute,
    "PanRightToolRelative": PanRightToolRelative,
    "PanRightToolAbsolute": PanRightToolAbsolute,
    "ZoomInToolRelative": ZoomInToolRelative,
    "ZoomInToolAbsolute": ZoomInToolAbsolute,
    "ZoomOutToolRelative": ZoomOutToolRelative,
    "ZoomOutToolAbsolute": ZoomOutToolAbsolute,
}
