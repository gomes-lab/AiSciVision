import os

import torch.nn as nn
import torch.optim as optim

import config
import utils
from dataloaders.datasets import dataset_name2ImageDatasetClass, get_dataloaders
from metrics import BinaryMetrics
from models.clip_classifier import CLIPMLPImageClassifier

# 1. Parse all experiment settings
args = utils.setup_clip_supervised_experiment()

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
num_labels = len(dataset_name2ImageDatasetClass[args.dataset_name].get_label_names())
clf = CLIPMLPImageClassifier(clip_model_config=config.clip_model_config, num_labels=num_labels)

# 5. Setup loss function, optimizer, and training loop
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.clip_mlp_model.mlp.parameters(), lr=args.lr)

if not args.no_train:
    train_metrics = BinaryMetrics()
    best_val_acc = float("-inf")
    for epoch in range(args.num_epochs):
        clf.train_epoch(train_dataloader, train_metrics, loss_function, optimizer, epoch, args.device)

        # Evaluate the model and save the best model
        if args.use_val_dataset:
            val_metrics = BinaryMetrics()
            clf.evaluate(val_dataloader, val_metrics, loss_function, device=args.device)
            curr_val_acc = val_metrics.accuracy.compute().item()
            if curr_val_acc > best_val_acc:
                print("Validation accuracy improved, saving best model...")
                best_val_acc = curr_val_acc
                val_metrics.save_metrics(os.path.join(args.log_folder, "best_val_metrics.json"))

                clf.save_model(os.path.join(args.log_folder, "best_model.pth"))

    clf.save_model(os.path.join(args.log_folder, "last_model.pth"))
    if not args.use_val_dataset:
        # Save the last model as best model since no validation was done
        clf.save_model(os.path.join(args.log_folder, "best_model.pth"))
    print(f"last {train_metrics=}")

    # 6. Evaluate the model
    if args.use_val_dataset:
        val_metrics = BinaryMetrics()
        clf.evaluate(val_dataloader, val_metrics, loss_function, device=args.device)
        print(f"last {val_metrics=}")
        val_metrics.save_metrics(os.path.join(args.log_folder, "val_metrics.json"))

    # Remove the classifier from memory
    del clf

# 7. Test the model
if args.evaluate_on_test:
    best_model_path = os.path.join(args.log_folder, "best_model.pth")
    print(f"Loading the best model from {best_model_path}")
    clf = CLIPMLPImageClassifier.load_model(
        best_model_path, model_init_kwargs=dict(clip_model_config=config.clip_model_config, num_labels=num_labels)
    )

    test_metrics = BinaryMetrics()
    clf.evaluate(test_dataloader, test_metrics, loss_function, args.number_of_test_samples, device=args.device)
    print(f"{test_metrics=}")
    test_metrics.save_metrics(os.path.join(args.log_folder, "test_metrics.json"))
