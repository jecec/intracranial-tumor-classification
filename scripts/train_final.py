from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassCohenKappa
)
from torchmetrics import MetricCollection
from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from scripts.utils import save_metrics_pkl
from utils import print_metrics

args = get_args()
device = args.device


def train_final(model, train_loader, checkpoint=None):
    """Training function for final model

    Args:
        model: Neural network model
        train_loader: Training data loader
        checkpoint: Optional checkpoint to resume from

    Returns:
        history: Dictionary containing training history
    Outputs:
        checkpoint.pth: Checkpoint with training history and states for model and optimizer
        final_model.pth: Final model weights after training
    """
    # TODO: Implement early stopping
    # TODO: Add more hyperparameters and training options here, such as batch normalization, learning rate scheduler etc.

    # Initialize training metrics using MetricCollection
    train_metrics_tracker = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=args.num_classes),
        'balanced_accuracy': MulticlassAccuracy(num_classes=args.num_classes, average='macro'),
        'precision': MulticlassPrecision(num_classes=args.num_classes, average='macro'),
        'recall': MulticlassRecall(num_classes=args.num_classes, average='macro'),
        'cohen_kappa': MulticlassCohenKappa(num_classes=args.num_classes),
    }).to(device)


    history = {
        "train_loss": [],
        "train_bac": [],
    }
    starting_epoch = 0

    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Load state from checkpoint if resuming
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        train_metrics_computed = checkpoint["train_metrics_computed"]

    # Main training loop
    for epoch in tqdm(range(starting_epoch, args.epochs), desc=f"Training"):
        model.train()
        train_loss = 0

        # Reset all metrics from tracker
        train_metrics_tracker.reset()

        for batch in train_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)

            optimizer.zero_grad()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update all metrics at once
            preds = outputs.argmax(dim=1)
            train_metrics_tracker.update(preds, targets)

        # Computing metrics collection
        train_metrics_computed = train_metrics_tracker.compute()
        train_metrics = {
            "loss": train_loss / len(train_loader),
            "accuracy": train_metrics_computed['accuracy'].item(),
            "balanced_accuracy": train_metrics_computed['balanced_accuracy'].item(),
            "precision": train_metrics_computed['precision'].item(),
            "recall": train_metrics_computed['recall'].item(),
            "cohen_kappa": train_metrics_computed['cohen_kappa'].item(),
        }

        # Logging history of metrics for plotting
        history["train_loss"].append(train_metrics["loss"])
        history["train_bac"].append(train_metrics["balanced_accuracy"])

        # Print metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print_metrics(train_metrics=train_metrics, epoch=epoch)

        # Save checkpoint every epoch
        checkpoint_data = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
            'train_metrics_computed': train_metrics_computed
        }
        torch.save(checkpoint_data, Path(args.checkpoint_dir, "checkpoint_main.pth"))

    # Save final model
    torch.save(model.state_dict(), f"{args.model_dir}/final_model.pth")
    save_metrics_pkl(train_metrics_computed)

    return history