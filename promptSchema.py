"""
Setting the PromptSchema requires the most specialization per new dataset and VisualRAG method.

The AiSciVision framework will interact with the LMM to make a classification in 3 stages:
1. It will pass a system prompt, then the image in question with the visual context (if any
is provided) and a prompt specified here, desribing the problem, and desribing the visual context.
2. The LMM is then allowed to use a list of tools - the tools and description of each is provided by
the prompt schema, for a set number of round (also set in the prompt schema).
3. The LMM must then make a final determinination of the class.
"""

import abc

import torch
from PIL import Image

from tools import Tool, aquaculture_tools, eelgrass_tools, solar_tools


class PromptSchema(abc.ABC):
    @abc.abstractmethod
    def __init__(self, tool_names: list[str], clip_path: str, num_tool_rounds: int = 4):
        """
        Initialize the PromptSchema.

        Args:
            tool_names (list of strings): A list of tools that the LMM can use.
            clip_path (str): The path to the CLIP model file.
            num_tool_rounds (int): The number of rounds for tool usage.
        """
        self.tool_names = tool_names
        self.num_tool_rounds = num_tool_rounds
        self.clip_path = clip_path

        self.tools: dict[str, Tool] = self.setup_tools(self.tool_names)

    ## Simple prompts and parsing
    @abc.abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abc.abstractmethod
    def get_final_prompt(self) -> str:
        pass

    @abc.abstractmethod
    def parse_final_answer(self, answer: str) -> float:
        pass

    ## Tool usage
    @abc.abstractmethod
    def get_tool_usage_prompt(self, round_num: int) -> str:
        """
        Get the prompt for tool usage.

        Args:
            round_num (int): The current round number.

        Returns:
            str: The prompt for tool usage.
        """
        pass

    @abc.abstractmethod
    def setup_tools(self, tool_names: list[str]) -> dict[str, Tool]:
        """
        Set up the tools for use.

        Args:
            tools (list of strings): A list of tools that the LMM can use.

        Returns:
            dict: A dictionary mapping tool names to tool objects.
        """
        pass

    @abc.abstractmethod
    def use_tool(self, tool_name: str, image: Image.Image) -> tuple[str, Image.Image | None]:
        """
        Use the specified tool on the given image.

        Args:
            tool_name (str): The name of the tool to use.
            image (PIL.Image): The image to apply the tool to.

        Returns:
            tuple: A tuple containing the text prompt and the resulting image (if any).
        """
        pass

    @abc.abstractmethod
    def get_supervised_tool_probability(self, image: Image.Image, md: dict) -> str:
        """
        Instantiate the Supervised ML Prediction Tool and return its output.

        Args:
            image (PIL.Image): The image to classify.
            md (dict): Metadata for the image.

        Returns:
            str: The prediction from the tool, written into text.
        """
        pass

    ## VisualRAG usage
    def _pos_neg_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for postive/negative RAG.

        Args:
            vis_context (dict[str, torch.Tensor]): Output of `VisualRAG.get_context`.
            image_to_classify (torch.Tensor): Target image.

        Returns:
            A list of tuples: first entry is prompt, and second entry is the image embedding from visualRAG.
        """
        pass

    @abc.abstractmethod
    def _no_context_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for no context.
        """
        pass

    def get_visual_context_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        This is called when we have visual context to show the LLM.
        It should return a list of tuples, where each tuple contains a prompt to send, and an image (image can be None).

        The LMM will then be prompted with each, and can allowed to respond, then prompted again with the next.

        Args:
            visual_context: a dictionary with identifiers for the type of context, and the image.
            image_to_classify: the image we want to classify.

        Returns:
            a list of tuples, where each tuple contains a prompt to send, and an image.
        """
        if "positiveExample" in visualRAG_context:
            return self._pos_neg_prompt(visualRAG_context, image_to_classify)
        elif "noContext" in visualRAG_context:
            return self._no_context_prompt(visualRAG_context, image_to_classify)
        else:
            raise NotImplementedError("Prompt schema not implemented for this visual context")


class EelgrassPromptSchema(PromptSchema):
    """
    Prompt schema for the eelgrass wasting disease classification task.
    """

    def __init__(self, tool_names: list[str], clip_path: str, num_tool_rounds: int = 4):
        super().__init__(tool_names, clip_path, num_tool_rounds)

    def get_system_prompt(self) -> str:
        return "You are an expert marine biologist specializing in eelgrass wasting disease classification. Your task is to determine whether an scan of an eelgrass blade contains eelgrass wasting disease or not. Eelgrass wasting disease, primarily caused by the marine pathogen Labyrinthula zosterae, presents visually as dark, necrotic lesions on the leaves of eelgrass (Zostera marina). These lesions typically appear as irregular black or brown spots or streaks, often starting at the tips of the leaves and spreading downwards. Over time, these lesions can coalesce, leading to extensive damage, including: 1. Spotting and Streaking: Initial symptoms include small, dark spots or streaks on the eelgrass leaves. 2. Blotching: As the disease progresses, these spots merge into larger blotches, which may appear dark brown or black. 3. Leaf Thinning and Loss: Affected areas may thin out or deteriorate, causing leaves to become more fragile and eventually break off. 4. Overall Decline: Heavily infected eelgrass beds often exhibit widespread dieback, with leaves appearing discolored, tattered, or decomposed.Use your knowledge and the tools provided to make an accurate classification."

    def get_final_prompt(self) -> str:
        return "Based on your analysis and the tools used, please provide your final classification regarding the presence of eelgrass wasting disease in the image. Provide a brief explanation for your decision, including key observations and any tool results that influenced your conclusion. Consider the color, texture, and pattern of the eelgrass leaves, as well as any visible lesions or discoloration that might indicate disease. At the end of your response, you must provide your classification in the following format: [Yes:PercentYes,No:PercentNo], where PercentYes is the probability you think eelgrass wasting disease is present, and PercentNo is the probability you think it is not present. These percentages must sum to 100%. To break ties, always assign at least 51% to one category. For example, [Yes:75,No:25] indicates 75% certainty of eelgrass wasting disease presence, while [Yes:49,No:51] indicates 51% certainty of its absence."

    def parse_final_answer(self, answer: str) -> float:
        try:
            # Find the last occurrence of '[' and ']' in the answer
            start = answer.rfind("[")
            end = answer.rfind("]")

            if start != -1 and end != -1 and start < end:
                # Extract the content between the last pair of brackets
                classification = answer[start + 1 : end]
                # Split the classification into Yes and No parts
                yes_part, no_part = classification.split(",")
                # Extract the percentage for Yes
                yes_percentage = float(yes_part.split(":")[1])
                # Convert percentage to a value between 0 and 1
                return yes_percentage / 100.0
            else:
                raise ValueError("No classification found in brackets")
        except ValueError:
            raise ValueError("Unable to parse classification percentages from LMM response")

    def setup_tools(self, tool_names: list[str]) -> dict[str, Tool]:
        tool_map = eelgrass_tools.tool_name2Tool_cls
        return {
            name: (tool_map[name](self.clip_path) if "predict" in name.lower() else tool_map[name]())
            for name in tool_names
        }

    def get_tool_usage_prompt(self, round_num: int) -> str:
        tool_descriptions = "\n".join([f"- {tool_name}: {tool.description}" for tool_name, tool in self.tools.items()])
        prompt = f"""You have access to the following tools to assist in your analysis:
{tool_descriptions}

You have {self.num_tool_rounds-round_num} opportunities to use these tools. You are encouraged to use at least 3 tools to gather sufficient information. While it's good to conclude early if you're certain about the classification, the most important thing is making the correct prediction. Feel free to use more tools if you need to increase your confidence.

If you want to use a tool, explain how it will assist your analysis, then respond with the tool name in square brackets, like this: [ToolName], or if you are finished, respond with [Finished].

Remember, your goal is to accurately classify whether the image shows eelgrass wasting disease or not. If you do use a tool, you will see the result of the tool immediately, and you should briefly, 1-2 sentences describe any new insights from the tool, and your thoughts on the classification.

After each tool use, end your thought with brackets in the format {{Yes:yesPercent,No:noPercent}} to indicate your current confidence/prediction. For example, {{Yes:70,No:30}} would indicate 70% confidence in the presence of eelgrass wasting disease and 30% confidence in its absence. Note that this is not your final prediction.

Then you will be asked again if you want to use a tool or are finished."""

        return prompt

    def use_tool(
        self, tool_name: str, image: Image.Image, round_num: int = -1, md: dict = {}
    ) -> tuple[str, Image.Image | None, dict]:
        assert tool_name in self.tools.keys()

        prompt, result_image, _ = self.tools[tool_name](image, md)  # ignore metadata returned by eelgrass tool
        rounds_left = self.num_tool_rounds - round_num
        if round_num == -1:
            return (prompt, result_image, {})
        else:
            prompt += f"\n\nYou have {rounds_left} rounds left. You may now choose another tool or indicate you're finished. Respond with ONLY [ToolName] to use a single tool, or [Finished] if you're done. Do not include multiple tool names or any other text in brackets. If you want to reference the last tool you used, simply write its name without brackets. After choosing a tool or indicating you're finished, provide your confidence in the presence of eelgrass wasting disease in the format {{yes:PercentYes,no:PercentNo}}. Then, explain why you are picking this tool or finishing, and how it relates to your current assessment of the image."
            return (prompt, result_image, md)

    def get_supervised_tool_probability(self, image: Image.Image, md: dict) -> str:
        predict_tool = eelgrass_tools.PredictEelgrassWastingDiseaseTool(self.clip_path)
        return predict_tool(image, md)

    def _pos_neg_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for postive/negative RAG.

        Args:
            vis_context (dict[str, torch.Tensor]): Output of `VisualRAG.get_context`.
            image_to_classify (torch.Tensor): Target image.

        Returns:
            A list of tuples: first entry is prompt, and second entry is the image embedding from visualRAG.
        """
        prompts_to_return = []

        pre_prompt = "I will start by first showing you an example image, that is visually similar to the image we want to classify, that does have eelgrass wasting disease. Then I will show you another example image that does not have eelgrass wasting disease, but is visually similar to the image we want to classify. Then I will show you the image we want to classify. For the first two, you should briefly (1-2 sentences) explain why it does or does not have eelgrass wasting disease. For the final image you should explain in a paragraph why you believe it does or does not have eelgrass wasting disease. \n\n"

        # Setup the positive prompt
        first_prompt = (
            pre_prompt
            + "\nHere is the example that does have eelgrass wasting disease, only respond describing it, nothing else:"
        )
        prompts_to_return.append((first_prompt, visualRAG_context["positiveExample"]))

        # Setup second prompt
        second_prompt = (
            "Here is the example that does not have eelgrass wasting disease only respond describing it, nothing else:"
        )
        prompts_to_return.append((second_prompt, visualRAG_context["negativeExample"]))

        # Setup third prompt
        third_prompt = "Here is the image need to classify as having eelgrass wasting disease or not. Describe what you see and compare and contrast it with the previous two known examples. Explain your thought process for classifying this image."
        prompts_to_return.append((third_prompt, image_to_classify))

        return prompts_to_return

    def _no_context_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for no context.
        """
        return [
            (
                "Here is the image we want to classify. Explain your thought process for classifying this image.",
                image_to_classify,
            )
        ]


class SolarPromptSchema(PromptSchema):
    """
    Prompt schema for the solar panel defect classification task.
    """

    def __init__(self, tool_names: list[str], clip_path: str, num_tool_rounds: int = 4):
        super().__init__(tool_names, clip_path, num_tool_rounds)

    def get_system_prompt(self) -> str:
        return "You are an expert in satellite imagery analysis specializing in solar panel detection. Your task is to determine whether a satellite image contains solar panels or not. Solar panels in satellite imagery typically appear as regular, geometric shapes with a distinct reflective surface. They are often arranged in arrays or grids on rooftops or in open fields. Key characteristics to look for include: 1. Geometric Patterns: Solar panels are usually arranged in rectangular or square shapes, forming distinct geometric patterns. 2. Reflectivity: Solar panels often appear brighter or more reflective than surrounding surfaces due to their glass or metal construction. 3. Color Contrast: Depending on the imagery, solar panels may appear darker or lighter than their surroundings, creating a noticeable contrast. 4. Location: Solar panels are commonly found on building rooftops, in open fields, or in designated solar farms. 5. Size and Scale: The size of solar panel arrays can vary greatly, from small rooftop installations to large utility-scale solar farms. Use your knowledge and the tools provided to make an accurate classification of whether solar panels are present in the given satellite image."

    def get_final_prompt(self) -> str:
        return "Based on your analysis and the tools used, please provide your final classification regarding the presence of solar panels in the satellite image. Provide a brief explanation for your decision, including key observations and any tool results that influenced your conclusion. Consider the geometric patterns, reflectivity, color contrast, location, and size of potential solar panel installations. At the end of your response, you must provide your classification in the following format: [Yes:PercentYes,No:PercentNo], where PercentYes is the probability you think solar panels are present, and PercentNo is the probability you think they are not present. These percentages must sum to 100%. To break ties, always assign at least 51% to one category. For example, [Yes:75,No:25] indicates 75% certainty of solar panel presence, while [Yes:49,No:51] indicates 51% certainty of their absence."

    def parse_final_answer(self, answer: str) -> float:
        try:
            # Find the last occurrence of '[' and ']' in the answer
            start = answer.rfind("[")
            end = answer.rfind("]")

            if start != -1 and end != -1 and start < end:
                # Extract the content between the last pair of brackets
                classification = answer[start + 1 : end]
                # Split the classification into Yes and No parts
                yes_part, no_part = classification.split(",")
                # Extract the percentage for Yes
                yes_percentage = float(yes_part.split(":")[1])
                # Convert percentage to a value between 0 and 1
                return yes_percentage / 100
            else:
                raise ValueError("No classification found in brackets")
        except ValueError:
            raise ValueError("Unable to parse classification percentages from LMM response")

    def setup_tools(self, tool_names: list[str]) -> dict[str, Tool]:
        tool_map = solar_tools.tool_name2Tool_cls
        return {
            name: (tool_map[name](self.clip_path) if "predict" in name.lower() else tool_map[name]())
            for name in tool_names
        }

    def get_tool_usage_prompt(self, round_num: int) -> str:
        tool_descriptions = "\n".join([f"- {tool_name}: {tool.description}" for tool_name, tool in self.tools.items()])
        prompt = f"""You have access to the following tools to assist in your analysis:
{tool_descriptions}

You have {self.num_tool_rounds-round_num} opportunities to use these tools. You are encouraged to use at least 3 tools to gather sufficient information. While it's good to conclude early if you're certain about the classification, the most important thing is making the correct prediction. Feel free to use more tools if you need to increase your confidence.

If you want to use a tool, explain how it will assist your analysis, then respond with the tool name in square brackets, like this: [ToolName], or if you are finished, respond with [Finished].

Remember, your goal is to accurately classify whether the image shows solar panels or not. If you do use a tool, you will see the result of the tool immediately, and you should briefly, 1-2 sentences describe any new insights from the tool, and your thoughts on the classification.

After each tool use, end your thought with brackets in the format {{Yes:yesPercent,No:noPercent}} to indicate your current confidence/prediction. For example, {{Yes:70,No:30}} would indicate 70% confidence in the presence of solar panels and 30% confidence in their absence. Note that this is not your final prediction.

Then you will be asked again if you want to use a tool or are finished."""

        return prompt

    def use_tool(
        self, tool_name: str, image: Image.Image, round_num: int = -1, md: dict = {}
    ) -> tuple[str, Image.Image | None, dict]:
        assert tool_name in self.tools.keys()

        prompt, result_image, _ = self.tools[tool_name](image, md)  # ignore metadata returned by solar tool
        rounds_left = self.num_tool_rounds - round_num
        if round_num == -1:
            return (prompt, result_image, {})
        else:
            prompt += f"\n\nYou have {rounds_left} rounds left. You may now choose another tool or indicate you're finished. Respond with ONLY [ToolName] to use a single tool, or [Finished] if you're done. Do not include multiple tool names or any other text in brackets. If you want to reference the last tool you used, simply write its name without brackets. After choosing a tool or indicating you're finished, provide your confidence in the presence of solar panels in the format {{yes:PercentYes,no:PercentNo}}. Then, explain why you are picking this tool or finishing, and how it relates to your current assessment of the image."
            return (prompt, result_image, md)

    def get_supervised_tool_probability(self, image: Image.Image, md: dict) -> str:
        predict_tool = solar_tools.PredictSolarPanelTool(self.clip_path)
        return predict_tool(image, md)

    def _pos_neg_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for postive/negative RAG.

        Args:
            vis_context (dict[str, torch.Tensor]): Output of `VisualRAG.get_context`.
            image_to_classify (torch.Tensor): Target image.

        Returns:
            A list of tuples: first entry is prompt, and second entry is the image embedding from visualRAG.
        """
        prompts_to_return = []

        # Setup first prompt
        first_prompt = "This is an example of a satellite image containing solar panels. Describe what you see, focusing on the characteristics that indicate the presence of solar panels."
        prompts_to_return.append((first_prompt, visualRAG_context["positiveExample"]))

        # Setup second prompt
        second_prompt = "This is an example of a satellite image without solar panels. Describe what you see, noting the absence of solar panels and any other relevant features."
        prompts_to_return.append((second_prompt, visualRAG_context["negativeExample"]))

        # Setup third prompt
        third_prompt = "Here is the satellite image we need to classify as having solar panels or not. Describe what you see and compare and contrast it with the previous two known examples. Explain your thought process for classifying this image."
        prompts_to_return.append((third_prompt, image_to_classify))

        return prompts_to_return

    def _no_context_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for no context.
        """
        return [
            (
                "Here is the satellite image we want to classify. Explain your thought process for determining if there are solar panels present in this image.",
                image_to_classify,
            )
        ]


class AquaculturePromptSchema(PromptSchema):
    """
    Prompt schema for the aquaculture pond detection classification task.
    """

    def __init__(self, tool_names: list[str], clip_path: str, num_tool_rounds: int = 4):
        super().__init__(tool_names, clip_path, num_tool_rounds)

    def get_system_prompt(self) -> str:
        return "You are an expert in satellite imagery analysis specializing in aquaculture pond detection. Your task is to determine whether a satellite image contains aquaculture ponds or not. Aquaculture ponds in satellite imagery typically appear as regular, geometric shapes with distinct water surfaces. They are often arranged in clusters or grids in coastal areas or inland water bodies. Key characteristics to look for include: 1. Geometric Patterns: Aquaculture ponds are usually rectangular or square in shape, forming distinct geometric patterns. 2. Water Color: The water in aquaculture ponds often has a different color or tone compared to natural water bodies, due to the high density of organisms and potential algae growth. 3. Surrounding Features: Look for access roads, feeding platforms, or other infrastructure associated with aquaculture operations. 4. Location: Aquaculture ponds are commonly found in coastal areas, estuaries, or near rivers and lakes. 5. Size and Scale: The size of aquaculture ponds can vary, but they are typically uniform within a single farm and arranged in an organized manner. 6. Texture: The water surface of aquaculture ponds often appears smoother than natural water bodies. Use your knowledge and the tools provided to make an accurate classification of whether aquaculture ponds are present in the given satellite image."

    def get_final_prompt(self) -> str:
        return "Based on your analysis and the tools used, please provide your final classification regarding the presence of an aquaculture pond in the satellite image. Provide a brief explanation for your decision, including key observations and any tool results that influenced your conclusion. Consider the shape, color, and arrangement of water bodies, as well as any surrounding infrastructure that might indicate aquaculture activity. At the end of your response, you must provide your classification in the following format: [Yes:PercentYes,No:PercentNo], where PercentYes is the probability you think it is an aquaculture pond, and PercentNo is the probability you think it is not an aquaculture pond. These percentages must sum to 100%. To break ties, always assign at least 51% to one category. For example, [Yes:75,No:25] indicates 75% certainty of an aquaculture pond's presence, while [Yes:49,No:51] indicates 51% certainty of its absence."

    def parse_final_answer(self, answer: str) -> float:
        try:
            # Find the last occurrence of '[' and ']' in the answer
            start = answer.rfind("[")
            end = answer.rfind("]")

            if start != -1 and end != -1 and start < end:
                # Extract the content between the last pair of brackets
                classification = answer[start + 1 : end]
                # Split the classification into Yes and No parts
                yes_part, no_part = classification.split(",")
                # Extract the percentage for Yes
                yes_percentage = float(yes_part.split(":")[1])
                # Convert percentage to a value between 0 and 1
                return yes_percentage / 100
            else:
                raise ValueError("No classification found in brackets")
        except ValueError:
            raise ValueError("Unable to parse classification percentages from LMM response")

    def setup_tools(self, tool_names: list[str]) -> dict[str, Tool]:
        tool_map = aquaculture_tools.tool_name2Tool_cls
        return {
            name: (tool_map[name](self.clip_path) if "predict" in name.lower() else tool_map[name]())
            for name in tool_names
        }

    def get_tool_usage_prompt(self, round_num: int) -> str:
        tool_descriptions = "\n".join([f"- {tool_name}: {tool.description}" for tool_name, tool in self.tools.items()])
        prompt = f"""You have access to the following tools to assist in your analysis:
{tool_descriptions}

You have {self.num_tool_rounds-round_num} opportunities to use these tools. You are encouraged to use at least 3 tools to gather sufficient information. While it's good to conclude early if you're certain about the classification, the most important thing is making the correct prediction. Feel free to use more tools if you need to increase your confidence.

If you want to use a tool, explain how it will assist your analysis, then respond with the tool name in square brackets, like this: [ToolName], or if you are finished, respond with [Finished].

Remember, your goal is to accurately classify whether the image shows aquaculture ponds or not. If you do use a tool, you will see the result of the tool immediately, and you should briefly, 1-2 sentences describe any new insights from the tool, and your thoughts on the classification.

After each tool use, end your thought with brackets in the format {{Yes:yesPercent,No:noPercent}} to indicate your current confidence/prediction. For example, {{Yes:70,No:30}} would indicate 70% confidence in the presence of aquaculture ponds and 30% confidence in their absence. Note that this is not your final prediction.

Then you will be asked again if you want to use a tool or are finished."""

        return prompt

    def use_tool(
        self, tool_name: str, image: Image.Image, round_num: int = -1, md: dict = {}
    ) -> tuple[str, Image.Image | None, dict]:
        assert tool_name in self.tools.keys()

        prompt, result_image, updated_md = self.tools[tool_name](
            image, md
        )  # ignore metadata returned by aquaculture tool
        rounds_left = self.num_tool_rounds - round_num
        if round_num == -1:
            return (prompt, result_image, updated_md)
        else:
            prompt += f"\n\nYou have {rounds_left} rounds left. You may now choose another tool or indicate you're finished. Respond with ONLY [ToolName] to use a single tool, or [Finished] if you're done. Do not include multiple tool names or any other text in brackets. If you want to reference the last tool you used, simply write its name without brackets. After choosing a tool or indicating you're finished, provide your confidence in the presence of aquaculture ponds in the format {{yes:PercentYes,no:PercentNo}}. Then, explain why you are picking this tool or finishing, and how it relates to your current assessment of the image."
            return (prompt, result_image, updated_md)

    def get_supervised_tool_probability(self, image: Image.Image, md: dict) -> str:
        predict_tool = aquaculture_tools.PredictAquaculturePondTool(self.clip_path)
        return predict_tool(image, md)

    def _pos_neg_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for postive/negative RAG.

        Args:
            vis_context (dict[str, torch.Tensor]): Output of `VisualRAG.get_context`.
            image_to_classify (torch.Tensor): Target image.

        Returns:
            A list of tuples: first entry is prompt, and second entry is the image embedding from visualRAG.
        """
        prompts_to_return = []

        # Setup first prompt
        first_prompt = "This is an example of a satellite image with an aquaculture pond. Describe what you see, noting the characteristics that identify it as an aquaculture pond."
        prompts_to_return.append((first_prompt, visualRAG_context["positiveExample"]))

        # Setup second prompt
        second_prompt = "This is an example of a satellite image without an aquaculture pond. Describe what you see, noting the absence of aquaculture ponds and any other relevant features."
        prompts_to_return.append((second_prompt, visualRAG_context["negativeExample"]))

        # Setup third prompt
        third_prompt = "Here is the satellite image we need to classify as having an aquaculture pond or not. Describe what you see and compare and contrast it with the previous two known examples. Explain your thought process for classifying this image."
        prompts_to_return.append((third_prompt, image_to_classify))

        return prompts_to_return

    def _no_context_prompt(
        self,
        visualRAG_context: dict[str, torch.Tensor],
        image_to_classify: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Prompt schema implementation for no context.
        """
        return [
            (
                "Here is the satellite image we want to classify. Explain your thought process for determining if there is an aquaculture pond present in this image.",
                image_to_classify,
            )
        ]


dataset_name2PromptSchema_cls: dict[str, PromptSchema] = {
    "aquaculture": AquaculturePromptSchema,
    "eelgrass": EelgrassPromptSchema,
    "solar": SolarPromptSchema,
}


def get_prompt_schema(dataset_name: str, tool_names: list[str], clip_path: str, num_tool_rounds: int) -> PromptSchema:
    """
    Factory function to get the appropriate PromptSchema based on the name.

    Args:
        dataset_name (str): The name of the dataset to use.
        tool_names (list of strings): A list of tools that the LMM can use.
        clip_path (str): Path to the clip model to use as the supervised model.
        num_tool_rounds (int): The number of rounds for tool usage.

    Returns:
        PromptSchema: An instance of the appropriate PromptSchema subclass.
    """
    assert dataset_name in dataset_name2PromptSchema_cls.keys()

    promptSchema_cls = dataset_name2PromptSchema_cls[dataset_name]
    return promptSchema_cls(tool_names, clip_path, num_tool_rounds)
