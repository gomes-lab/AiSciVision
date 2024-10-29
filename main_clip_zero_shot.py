import json
import os

from sklearn.metrics import roc_auc_score

import config
import utils
from dataloaders.datasets import dataset_name2ImageDatasetClass, get_dataloaders
from metrics import BinaryMetrics
from models.clip_classifier import CLIPZeroShotImageClassifier


def calculate_summary_stats(results: dict[str, list], save_file_name: str) -> None:
    clip_zero_preds = [pred["class"] for pred in results["clip_zero_preds"]]
    clip_zero_probs = [pred["probability"] for pred in results["clip_zero_preds"]]
    ground_truths = results["ground_truths"]

    # Redo some calculations as BinaryMetrics, just so everything is saved in this file
    clip_zero_accuracy = utils.compute_accuracy(clip_zero_preds, ground_truths)
    clip_zero_f1 = utils.compute_f1(clip_zero_preds, ground_truths)
    clip_zero_precision = utils.compute_precision(clip_zero_preds, ground_truths)
    clip_zero_recall = utils.compute_recall(clip_zero_preds, ground_truths)

    # Handle the case where there's only one class for ROC AUC calculation
    if len(set(ground_truths)) == 1:
        clip_zero_roc_auc = -1
    else:
        clip_zero_roc_auc = roc_auc_score(ground_truths, clip_zero_probs)

    total_images = len(results["ids"])

    # Create summary statistics dictionary
    sum_stats = {
        "ids": results["ids"],
        "clip_zero_preds": results["clip_zero_preds"],
        "ground_truths": ground_truths,
        "total_looked_at": total_images,
        "clip_zero_accuracy": clip_zero_accuracy,
        "clip_zero_f1": clip_zero_f1,
        "clip_zero_precision": clip_zero_precision,
        "clip_zero_recall": clip_zero_recall,
        "clip_zero_roc_auc": clip_zero_roc_auc,
        "total_images": total_images,
    }

    # Save summary statistics
    with open(os.path.join(args.log_folder, save_file_name), "w") as f:
        json.dump(sum_stats, f, indent=4, sort_keys=True)

    # Print summary statistics
    print("\n--- Experiment Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"\nFinal CLIP Zero-Shot accuracy: {clip_zero_accuracy:.4f}")
    print(f"Final CLIP Zero-Shot F1 score: {clip_zero_f1:.4f}")
    print(f"Final CLIP Zero-Shot Precision: {clip_zero_precision:.4f}")
    print(f"Final CLIP Zero-Shot Recall: {clip_zero_recall:.4f}")
    print(f"Final CLIP Zero-Shot ROC AUC: {clip_zero_roc_auc:.4f}")
    print("--------------------------")


# 1. Parse all experiment settings
args = utils.setup_clip_zero_shot_experiment()

# 2. Set random seed for reproducibility
utils.seed_everything(args.seed)

# 3. Load the dataset
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
    args.data_root_dir,
    args.dataset_name,
    args.batch_size,
    args.percent_labeled,
    args.test_order_path,
    args.use_val_dataset,
)

# 4. Setup CLIP Zero shot classifier and fit on training set
label_names = [f"A photo of {label}" for label in dataset_name2ImageDatasetClass[args.dataset_name].get_label_names()]
model = CLIPZeroShotImageClassifier(clip_model_config=config.clip_model_config, label_names=label_names)
train_metrics = BinaryMetrics()
model.train_epoch(train_dataloader, train_metrics, device=args.device)

# 5. Evaluate the model
if args.use_val_dataset:
    val_metrics = BinaryMetrics()
    val_results = model.evaluate(val_dataloader, val_metrics, device=args.device)
    print(f"{val_metrics=}")
    val_metrics.save_metrics(os.path.join(args.log_folder, "val_metrics.json"))

    calculate_summary_stats(val_results, "val_sum_stats.json")

# 6. Test the model
if args.evaluate_on_test:
    test_metrics = BinaryMetrics()
    test_results = model.evaluate(test_dataloader, test_metrics, args.number_of_test_samples, device=args.device)
    print(f"{test_metrics=}")
    test_metrics.save_metrics(os.path.join(args.log_folder, "test_metrics.json"))

    calculate_summary_stats(test_results, "test_sum_stats.json")
