clip_model_config: str = "openai/clip-vit-base-patch32"

data_split_choices: list[str] = ["train", "valid", "test"]

val_dataset_frac: float = 0.1

dataset_names: list[str] = ["aquaculture", "eelgrass", "solar"]

dataset_name2tool_list: dict[str, list[str]] = {
    "aquaculture": "PredictAquaculturePondTool PanUpToolRelative PanUpToolAbsolute PanDownToolRelative PanDownToolAbsolute PanLeftToolRelative PanLeftToolAbsolute PanRightToolRelative PanRightToolAbsolute ZoomInToolRelative ZoomInToolAbsolute ZoomOutToolRelative ZoomOutToolAbsolute".split(
        " "
    ),
    "eelgrass": "PredictEelgrassWastingDiseaseTool HistogramEqualizationTool AdjustBrightnessTool SharpenTool EdgeDetectionTool IncreaseContrastTool DecreaseContrastTool".split(
        " "
    ),
    "solar": "PredictSolarPanelTool HistogramEqualizationTool AdjustBrightnessTool SharpenTool EdgeDetectionTool IncreaseContrastTool DecreaseContrastTool".split(
        " "
    ),
}
