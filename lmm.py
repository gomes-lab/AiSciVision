import abc
import base64
import io
import time

import openai
from PIL import Image

from aiSciVision import CONV_T


class LMM(abc.ABC):
    @abc.abstractmethod
    def process_conversation(self, conversation: CONV_T) -> str:
        pass


# {"type": "text", "text": "data..."}
OPENAI_MESS_TEXT_T = dict[str, str]

# When key is "type", value is "image_url".
# When key is "image_url", value is {"url": "data..."}
OPENAI_MESS_IMG_T = dict[str, str | dict[str, str]]

# {"role": "user", "content": [...]}
OPENAI_MESS_T = dict[str, str | list[OPENAI_MESS_TEXT_T | OPENAI_MESS_IMG_T]]


class GPT4Vision(LMM):
    """
    Assumes that the OpenAI API key is set as an environment variable `OPENAI_API_KEY`.
    """

    model_name: str = "gpt-4o"

    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0, seed: int = 1994, temperature: float = 0.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.seed = seed
        self.temperature = temperature

    def _encode_image(self, image: Image.Image) -> str:
        """Returns the base64 encoded of image."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_system_fingerprint(self) -> str:
        """
        Get the system fingerprint of the GPT-4 Vision model.

        Returns:
            str: The system fingerprint.
        """
        messages: list[OPENAI_MESS_T] = [{"role": "user", "content": "Hello"}]
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    seed=self.seed,
                )
                return response.system_fingerprint
            except (openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError) as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff

    def _call_api(self, messages: list[OPENAI_MESS_T]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0,
                    seed=self.seed,
                )
                return response.choices[0].message["content"]
            except (openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError) as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff

    def process_conversation(self, conversation: CONV_T) -> str:
        """
        Converts conversation into a list of messages that the API can parse, and calls the LMM API for a response.
        The response is returned.
        """
        # Transform conversation into messages that API can parse
        messages = []
        for entry in conversation:
            message: OPENAI_MESS_T = {"role": entry["role"], "content": []}

            # Take out text and image from the conversation, and parse it for API
            text, image = entry["message"]
            message["content"].append({"type": "text", "text": text})
            if image:
                message["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._encode_image(image)}"}}
                )

            messages.append(message)

        response = self._call_api(messages)
        return response


lmm_name2LMM_cls: dict[str, LMM] = {
    "gpt-4o": GPT4Vision,
}


def get_lmm(name: str, seed: int) -> LMM:
    assert name in lmm_name2LMM_cls.keys()

    lmm_cls = lmm_name2LMM_cls[name]
    return lmm_cls(seed=seed)
