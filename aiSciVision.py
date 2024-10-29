"""
AiSciVision orchestrates between the VisualRAG system and the tooling to track the conversation with the LMM.
"""

import torch
from PIL import Image

from promptSchema import PromptSchema
from visualRAG import VisualRAG

# {"role": "user", "content": ("text", optional "image")}
CONV_TURN_T = dict[str, str | tuple[str, Image.Image | None]]
CONV_T = list[CONV_TURN_T]


class AiSciVision:
    def __init__(self, vis_rag: VisualRAG, prompt_schema: PromptSchema) -> None:
        """
        Initialize the AiSciVision system.

        Args:
            vis_rag (VisualRAG): The Visual RAG system.
            prompt_schema (PromptSchema): The prompt schema (which contains the tools and the number of rounds of tool use)
        """
        self.vis_rag = vis_rag
        self.prompt_schema = prompt_schema
        self.conversation: list[CONV_T] = []

    def set_system_prompt(self) -> None:
        """
        Set the system prompt using the prompt schema.
        """
        system_prompt = self.prompt_schema.get_system_prompt()
        self.conversation.append({"role": "system", "message": (system_prompt, None)})

    def get_initial_prompts(self, image: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """
        Get the initial prompt using the visual RAG and prompt schema.

        Args:
            image: The input image to be classified (tensor of image needs to be unsqueezed so the batch size is 1).

        Returns:
            A list of tuples (prompt, image) for the initial prompts.
        """
        context = self.vis_rag.get_context(image)
        initial_prompts = self.prompt_schema.get_visual_context_prompt(context, image)
        return initial_prompts

    def update_conversation(self, role: str, message: str, image: torch.Tensor | None = None) -> None:
        """
        Update the conversation history with a new message.

        Args:
            role (str): The role of the sender ("system", "user", or "assistant").
            message (str): The text content of the message.
            image (torch.Tensor | None): The image tensor associated with the message, if any.
        """
        if image is not None:
            # Check if image is already a PIL Image, if not convert tensor to PIL Image
            if not isinstance(image, Image.Image):
                pil_image = Image.fromarray((image.squeeze().permute(1, 2, 0) * 255.0).byte().numpy())
            else:
                pil_image = image
        else:
            pil_image = None

        self.conversation.append({"role": role, "message": (message, pil_image)})

    def get_final_prompt(self) -> str:
        """
        Get the final prompt from the prompt schema.

        Returns:
            str: The final prompt for classification.
        """
        return self.prompt_schema.get_final_prompt()

    def parse_final_answer(self, answer: str) -> dict[str, float | int]:
        """
        Parse the final answer using the prompt schema.

        Args:
            answer (str): The LMM's final answer.

        Returns:
            dict with keys predicted "probability" and "class", extracted from the answer.
        """
        parsed_probability = self.prompt_schema.parse_final_answer(answer)
        classification = 1 if parsed_probability >= 0.5 else 0
        return {"probability": parsed_probability, "class": classification}
