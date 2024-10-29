"""
Tools are a way for the LMM to gain more information about the image in question.

Tools can be the output of another ML model, or can be determinstally computed metrics about the image in question,
or a way for the LMM to interact with the data.

All tools should be one of two types:
Type 1: Machine Learning inference - takes an image performs some ML algorithm on it, and returns the results as a string
- necessary to be a tool to save on cost of inference, and also LMM can act as orchestrator.

Type 2: Image manipulation - takes an image and performs some manipulation on it - move its for satellites, or
increases contrast or something like that etc.
"""

import abc

from PIL import Image


class Tool(abc.ABC):
    @abc.abstractmethod
    def __call__(self, image: Image.Image, md: dict, **kwargs) -> tuple[str, Image.Image | None, dict]:
        """
        Args:
            image (PIL.Image): The image to analyze.
            md (dict): Metadata about the image.

        Returns:
            tuple: A tuple containing the tool's analysis, the transformed image (if any), and metadata (dict could be empty).
        """
        pass
