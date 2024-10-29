import json
import os

from sklearn.metrics import roc_auc_score

import config
import embeddingModel
import utils
from dataloaders.datasets import get_dataloaders
from metrics import BinaryMetrics
from models.knn_classifier import KNNImageClassifier


def calculate_summary_stats(results: dict[str, list], save_file_name: str) -> None:
    knn_preds = [pred["class"] for pred in results["knn_preds"]]
    knn_probs = [pred["probability"] for pred in results["knn_preds"]]
    ground_truths = results["ground_truths"]

    # Redo some calculations as BinaryMetrics, just so everything is saved in this file
    knn_accuracy = utils.compute_accuracy(knn_preds, ground_truths)
    knn_f1 = utils.compute_f1(knn_preds, ground_truths)
    knn_precision = utils.compute_precision(knn_preds, ground_truths)
    knn_recall = utils.compute_recall(knn_preds, ground_truths)

    # Handle the case where there's only one class for ROC AUC calculation
    if len(set(ground_truths)) == 1:
        knn_roc_auc = -1
    else:
        knn_roc_auc = roc_auc_score(ground_truths, knn_probs)

    total_images = len(results["ids"])

    # Create summary statistics dictionary
    sum_stats = {
        "ids": results["ids"],
        "knn_preds": results["knn_preds"],
        "ground_truths": ground_truths,
        "total_looked_at": total_images,
        "knn_accuracy": knn_accuracy,
        "knn_f1": knn_f1,
        "knn_precision": knn_precision,
        "knn_recall": knn_recall,
        "knn_roc_auc": knn_roc_auc,
        "total_images": total_images,
    }

    # Save summary statistics
    with open(os.path.join(args.log_folder, save_file_name), "w") as f:
        json.dump(sum_stats, f, indent=4, sort_keys=True)

    # Print summary statistics
    print("\n--- Experiment Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"\nFinal KNN accuracy: {knn_accuracy:.4f}")
    print(f"Final KNN F1 score: {knn_f1:.4f}")
    print(f"Final KNN Precision: {knn_precision:.4f}")
    print(f"Final KNN Recall: {knn_recall:.4f}")
    print(f"Final KNN ROC AUC: {knn_roc_auc:.4f}")
    print("--------------------------")


# 1. Parse all experiment settings
args = utils.setup_knn_experiment()

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

# 4. Setup knn classifier and fit on training set
embedding_model_init_kwargs = {}
if args.embedding_model_name in ["clip", "clip_vision"]:
    embedding_model_init_kwargs = dict(clip_model_config=config.clip_model_config)
embedding_model = embeddingModel.get_embedding_model(args.embedding_model_name, embedding_model_init_kwargs)
clf = KNNImageClassifier(n_neighbors=args.n_neighbors, embedding_model=embedding_model)

train_metrics = BinaryMetrics()
clf.train_epoch(train_dataloader, train_metrics)
clf.save_model(os.path.join(args.log_folder, "model.pkl"))

# 5. Evaluate the model
if args.use_val_dataset:
    val_metrics = BinaryMetrics()
    val_results = clf.evaluate(val_dataloader, val_metrics)
    print(f"{val_metrics=}")
    val_metrics.save_metrics(os.path.join(args.log_folder, "val_metrics.json"))

    calculate_summary_stats(val_results, "val_sum_stats.json")

# 6. Test the model
if args.evaluate_on_test:
    test_metrics = BinaryMetrics()
    test_results = clf.evaluate(test_dataloader, test_metrics, args.number_of_test_samples)
    print(f"{test_metrics=}")
    test_metrics.save_metrics(os.path.join(args.log_folder, "test_metrics.json"))

    calculate_summary_stats(test_results, "test_sum_stats.json")
