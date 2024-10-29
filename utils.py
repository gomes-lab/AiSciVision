import argparse
import base64
import json
import os
import random
from io import BytesIO
from pathlib import Path

import geobench
import markdown
import numpy as np
import torch
from geobench.geobench_download import decompress_zip_with_progress
from huggingface_hub import hf_hub_download
from xhtml2pdf import pisa

import embeddingModel


def get_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AiSciVision experiments")
    parser.add_argument("--data_root_dir", type=str, default="Data/", help="Path to all datasets")
    parser.add_argument("--dataset_name", type=str, default="eelgrass", help="Which dataset to run on")
    parser.add_argument("--log_folder", type=str, default="logs/test_eelgrass_4o", help="Path to folder for logging")
    parser.add_argument("--percent_labeled", type=int, default=5, help="Percent of labeled data to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--seed", type=int, default=1994, help="Random seed for reproducibility")
    parser.add_argument("--use_cuda", action="store_true", help="Flag to use cuda if available")
    parser.add_argument("--use_val_dataset", action="store_true", help="Flag to use validation split")
    parser.add_argument("--evaluate_on_test", action="store_true", help="Flag to evaluate on test split")
    parser.add_argument("--number_of_test_samples", type=int, default=100, help="Number of test samples to process")
    parser.add_argument(
        "--test_order_path",
        type=str,
        default=None,
        help="Path to the test order directory  see create_test_set_selection.py (optional)",
    )

    return parser


def parse_all_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    # Create device arg
    args.device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"

    # Create logging folder
    os.makedirs(args.log_folder, exist_ok=True)

    # Save experiment settings
    settings_file_path = os.path.join(args.log_folder, "experiment_settings.json")
    print(f"* Saving experiment settings saved to {settings_file_path}...")
    with open(settings_file_path, "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    return args


def setup_experiment() -> argparse.Namespace:
    parser = get_common_parser()

    # Embedding model
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="clip_vision",
        choices=embeddingModel.embedding_name2model_cls.keys(),
        help="Which embedding model to use",
    )
    # VisualRAG arguments
    parser.add_argument(
        "--rag_type",
        type=str,
        default="PositiveNegative",
        choices=["PositiveNegative", "NoContext"],
        help="Type of Visual RAG to use",
    )
    # Add lmm name
    parser.add_argument(
        "--lmm_name", type=str, default="gpt-4o", choices=["gpt-4o", "gpt-4", "gpt-3.5"], help="Which LMM to use"
    )
    # PromptSchema arguments
    parser.add_argument(
        "--prompt_schema_name",
        type=str,
        default="eelgrass",
        choices=["eelgrass", "solar", "aquaculture"],
        help="Which prompt schema to use",
    )

    # Tools arguments
    parser.add_argument(
        "--tools",
        nargs="+",
        default=[
            "IncreaseContrastTool",
            "DecreaseContrastTool",
            "PredictEelgrassWastingDiseaseTool",
            "AdjustBrightnessTool",
            "SharpenTool",
            "EdgeDetectionTool",
            "HistogramEqualizationTool",
        ],
        choices=[
            "IncreaseContrastTool",
            "DecreaseContrastTool",
            "PredictEelgrassWastingDiseaseTool",
            "PredictSolarPanelTool",
            "PredictAquaculturePondTool",
            "PanUpToolRelative",
            "PanUpToolAbsolute",
            "PanDownToolRelative",
            "PanDownToolAbsolute",
            "PanLeftToolRelative",
            "PanLeftToolAbsolute",
            "PanRightToolRelative",
            "PanRightToolAbsolute",
            "ZoomInToolRelative",
            "ZoomInToolAbsolute",
            "ZoomOutToolRelative",
            "ZoomOutToolAbsolute",
            "AdjustBrightnessTool",
            "SharpenTool",
            "EdgeDetectionTool",
            "HistogramEqualizationTool",
        ],
        help="List of tools to use in the experiment",
    )
    # PromptSchema arguments
    parser.add_argument("--num_tool_rounds", type=int, default=4, help="Number of rounds for tool usage")

    # Add required argument for full path to CLIP model
    parser.add_argument("--clip_model_path", type=str, required=True, help="Full path to the CLIP model file")

    return parse_all_args(parser)


def setup_supervised_experiment() -> argparse.Namespace:
    parser = get_common_parser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes in the dataset")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save the trained model")

    return parse_all_args(parser)


def setup_knn_experiment() -> argparse.Namespace:
    parser = get_common_parser()
    parser.add_argument("--n_neighbors", type=int, required=True, help="Number of neighbors for the classifier")
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="mean",
        choices=embeddingModel.embedding_name2model_cls.keys(),
        help="Which embedding model to use",
    )

    return parse_all_args(parser)


def setup_clip_zero_shot_experiment() -> argparse.Namespace:
    parser = get_common_parser()

    return parse_all_args(parser)


def setup_clip_supervised_experiment() -> argparse.Namespace:
    parser = get_common_parser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no_train", action="store_true", help="Flag to skip training")

    return parse_all_args(parser)


def seed_everything(seed: int) -> None:
    """
    Set the seed for all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for reproducibility.")


def evaluate_final_answer(final_answer: int, label: int) -> None:
    """
    Evaluate the final answer against the true label and print the result.

    Args:
        final_answer (int): The predicted classification (0 or 1).
        label (int): The true label (0 or 1).
    """
    if final_answer == label:
        result = "Correct"
    else:
        result = "Incorrect"

    print(f"Prediction: {final_answer}, True Label: {label}, Result: {result}")


def create_conversation_pdf(
    conversation,
    output_path: str,
    true_class: int,
    lmm_prediction: int,
    lmm_probability: float,
    supervised_prediction: int,
    supervised_probability: float,
) -> None:
    """
    Create a PDF document from a conversation.

    Args:
        conversation (List[Dict]): A list of conversation entries, each containing 'role' and 'message'.
        output_path (str): The path where the PDF should be saved.
        true_class (int): The true class label.
        lmm_prediction (int): The prediction from the LMM model.
        lmm_probability (float): The probability of the LMM prediction.
        supervised_prediction (int): The prediction from the supervised model.
        supervised_probability (float): The probability of the supervised prediction.
    """

    # Convert markdown to HTML
    def markdown_to_html(text):
        return markdown.markdown(text)

    # Create HTML content
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .role {{ color: blue; font-size: 14px; font-weight: bold; }}
            .message {{ font-size: 12px; }}
            .header {{ color: green; font-size: 16px; }}
        </style>
    </head>
    <body>
        <div class="header">
            True Class: {true_class}, 
            LMM Prediction: {lmm_prediction} (Probability: {lmm_probability:.2f}), 
            Supervised Prediction: {supervised_prediction} (Probability: {supervised_probability:.2f})
        </div>
        <br>
    """

    for entry in conversation:
        role = entry["role"]
        message, image = entry["message"]

        html_content += f'<div class="role">{role.capitalize()}:</div>'
        html_content += f'<div class="message">{markdown_to_html(message)}</div>'

        if image:
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            html_content += f'<img src="data:image/png;base64,{img_base64}" style="width:400px;height:300px;">'

        html_content += "<br>"

    html_content += "</body></html>"

    # Convert HTML to PDF
    with open(output_path, "w+b") as output_file:
        pisa_status = pisa.CreatePDF(html_content, dest=output_file)

    # Ensure the file is saved
    if not os.path.exists(output_path) or pisa_status.err:
        raise FileNotFoundError(f"Failed to save conversation PDF to {output_path}")

    print(f"Conversation PDF saved to {output_path}")


def save_accuracy(parsed_final_answer: int, label: int, output_path: str) -> None:
    """
    Save the accuracy of a single prediction to a text file.

    Args:
        parsed_final_answer (int): The predicted label (0 or 1).
        label (int): The true label (0 or 1).
        output_path (str): The path where the accuracy file should be saved.
    """
    correct = int(parsed_final_answer == label)
    accuracy = 100 if correct else 0

    with open(output_path, "w") as f:
        f.write(f"Predicted: {parsed_final_answer}\n")
        f.write(f"Actual: {label}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy}%\n")

    print(f"Accuracy saved to {output_path}")


def compute_accuracy(predictions, ground_truths):
    """
    Compute the accuracy of predictions compared to ground truths.

    Args:
        predictions (list): List of predicted labels.
        ground_truths (list): List of true labels.

    Returns:
        float: The accuracy score.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Length of predictions and ground truths must be the same.")

    correct = sum(pred == gt for pred, gt in zip(predictions, ground_truths))
    return correct / len(predictions)


def compute_f1(predictions, ground_truths):
    """
    Compute the F1 score of predictions compared to ground truths.

    Args:
        predictions (list): List of predicted labels.
        ground_truths (list): List of true labels.

    Returns:
        float: The F1 score.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Length of predictions and ground truths must be the same.")

    true_positives = sum((pred == 1 and gt == 1) for pred, gt in zip(predictions, ground_truths))
    false_positives = sum((pred == 1 and gt == 0) for pred, gt in zip(predictions, ground_truths))
    false_negatives = sum((pred == 0 and gt == 1) for pred, gt in zip(predictions, ground_truths))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def compute_precision(predictions, ground_truths):
    """
    Compute the precision of predictions compared to ground truths.

    Args:
        predictions (list): List of predicted labels.
        ground_truths (list): List of true labels.

    Returns:
        float: The precision score.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Length of predictions and ground truths must be the same.")

    true_positives = sum((pred == 1 and gt == 1) for pred, gt in zip(predictions, ground_truths))
    false_positives = sum((pred == 1 and gt == 0) for pred, gt in zip(predictions, ground_truths))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def compute_recall(predictions, ground_truths):
    """
    Compute the recall of predictions compared to ground truths.

    Args:
        predictions (list): List of predicted labels.
        ground_truths (list): List of true labels.

    Returns:
        float: The recall score.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Length of predictions and ground truths must be the same.")

    true_positives = sum((pred == 1 and gt == 1) for pred, gt in zip(predictions, ground_truths))
    false_negatives = sum((pred == 0 and gt == 1) for pred, gt in zip(predictions, ground_truths))

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def download_geobench_tasks(task_hf_paths: list[str]) -> None:
    """
    Download the specified list of tasks.
    `geobench-download` would download everything, and this function only does what's needed.
    Implementation adapted from https://github.com/ServiceNow/geo-bench/blob/main/geobench/geobench_download.py

    Args:
        task_names (list[str]): List of task names, e.g. `["classification_v1.0/m-pv4ger.zip"]`.
    """
    local_directory = Path(geobench.GEO_BENCH_DIR)
    dataset_repo = "recursix/geo-bench-1.0"

    local_directory.mkdir(parents=True, exist_ok=True)

    for file in task_hf_paths:
        local_file_path = local_directory / file

        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {file}...")
        hf_hub_download(
            repo_id=dataset_repo,
            filename=file,
            cache_dir=local_directory,
            local_dir=local_directory,
            repo_type="dataset",
        )

    # Decompress each file sequentially
    zip_files = [file for file in task_hf_paths if file.endswith(".zip")]

    for i, zip_file in enumerate(zip_files):
        print(f"Decompressing {i+1}/{len(zip_files)}: {zip_file}  ...")
        decompress_zip_with_progress(local_directory / zip_file)

    print("Download and decompression process completed.")
