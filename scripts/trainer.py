from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
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


def train(model, train_loader, val_loader, fold, checkpoint=None, all_fold_metrics=None):
    """Training function for K-fold cross-validation

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        fold: Current fold number
        checkpoint: Optional checkpoint to resume from (should be from the SAME fold)
        all_fold_metrics: List of metrics from all completed folds (for checkpoint persistence)

    Returns:
        history: Dictionary containing training history for this fold
        best_metrics: Best validation metrics achieved during training for this fold
    """
    # TODO: Implement early stopping
    # TODO: Add more hyperparameters and training options here, such as regularization, learning rate scheduler etc.

    # Initialize training metrics using MetricCollection
    train_metrics_tracker = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=args.num_classes, average='micro'),
        'precision': MulticlassPrecision(num_classes=args.num_classes, average='macro'),
        'recall': MulticlassRecall(num_classes=args.num_classes, average='macro'),
        'cohen_kappa': MulticlassCohenKappa(num_classes=args.num_classes),
    }).to(device)

    # Initialize history and metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recall": [],
        "val_recall": [],
    }
    best_metrics = {}
    best_bac = 0.0
    starting_epoch = 0

    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Load state from checkpoint if resuming
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        best_metrics = checkpoint["best_metrics"]
        best_bac = checkpoint.get("best_bac", 0.0)

    for epoch in tqdm(range(starting_epoch, args.epochs), desc=f"Training Fold {fold + 1}"):
        model.train()
        train_loss = 0

        # Reset all metrics
        train_metrics_tracker.reset()

        for batch in train_loader:
            # Load batch data
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)

            # Resetting the gradients
            optimizer.zero_grad()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update all metrics at once
            preds = outputs.argmax(dim=1)
            train_metrics_tracker.update(preds, targets)

        train_metrics_computed = train_metrics_tracker.compute()
        train_metrics = {
            "loss": train_loss / len(train_loader),
            "accuracy": train_metrics_computed['accuracy'].item(),
            "precision": train_metrics_computed['precision'].item(),
            "recall": train_metrics_computed['recall'].item(),
            "cohen_kappa": train_metrics_computed['cohen_kappa'].item(),
        }

        # Validation Metrics
        val_metrics = validate(model, val_loader, criterion)

        # Logging metrics for plotting
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

        # Print metrics every {args.print_rate} epochs
        if (epoch + 1) % args.print_rate == 0:
            print_metrics(train_metrics=train_metrics, eval_metrics=val_metrics, epoch=epoch, fold=fold)

        # Save checkpoint every epoch
        checkpoint_data = {
            'epoch': epoch,
            'fold': fold,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
            'best_metrics': best_metrics,
            'best_bac': best_bac,
            'all_fold_metrics': all_fold_metrics,  # Store ALL fold metrics
        }
        torch.save(checkpoint_data, Path(args.checkpoint_dir, "checkpoint_cv.pth"))

        # Save best model based on macro recall score
        if val_metrics["recall"] > best_bac:
            best_bac = val_metrics["recall"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), f"{args.model_dir}/best_model_fold_{fold + 1}.pth")
            save_metrics_pkl(best_metrics, "validate_kfold", fold)

    return history, best_metrics


def validate(model, val_loader, criterion):
    """Validation function for computing metrics on validation set

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function

    Returns:
        metrics: Dictionary of validation metrics
    """
    # Initialize validation metrics using MetricCollection
    val_metrics_tracker = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=args.num_classes, average='micro'),
        'precision': MulticlassPrecision(num_classes=args.num_classes, average='macro'),
        'recall': MulticlassRecall(num_classes=args.num_classes, average='macro'),
        'f1_macro': MulticlassF1Score(num_classes=args.num_classes, average='macro'),
        'cohen_kappa': MulticlassCohenKappa(num_classes=args.num_classes),
        'confusion_matrix': MulticlassConfusionMatrix(num_classes=args.num_classes),
    }).to(device)

    # Separate metrics that need different inputs (probabilities vs predictions)
    roc_auc = MulticlassAUROC(num_classes=args.num_classes, average='macro').to(device)
    f1_per_class = MulticlassF1Score(num_classes=args.num_classes, average=None).to(device)
    precision_per_class = MulticlassPrecision(num_classes=args.num_classes, average=None).to(device)
    recall_per_class = MulticlassRecall(num_classes=args.num_classes, average=None).to(device)

    val_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Update metrics
            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)

            # Update all prediction-based metrics at once
            val_metrics_tracker.update(preds, targets)

            # Update probability-based and per-class metrics separately
            roc_auc.update(probs, targets)
            f1_per_class.update(preds, targets)
            precision_per_class.update(preds, targets)
            recall_per_class.update(preds, targets)

    # Compute all metrics
    val_metrics_computed = val_metrics_tracker.compute()
    metrics = {
        "loss": val_loss / len(val_loader),
        "accuracy": val_metrics_computed['accuracy'].item(),
        "precision": val_metrics_computed['precision'].item(),
        "recall": val_metrics_computed['recall'].item(),
        "macro_f1": val_metrics_computed['f1_macro'].item(),
        "cohen_kappa": val_metrics_computed['cohen_kappa'].item(),
        "roc_auc_macro": roc_auc.compute().item(),
        "per_label_f1": f1_per_class.compute().cpu().numpy(),
        "per_label_precision": precision_per_class.compute().cpu().numpy(),
        "per_label_recall": recall_per_class.compute().cpu().numpy(),
        "confusion_matrix": val_metrics_computed['confusion_matrix'].cpu().numpy(),
    }

    return metrics