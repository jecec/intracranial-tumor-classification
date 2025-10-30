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

from utils import print_metrics
args = get_args()
device = args.device

def train(model, train_loader, val_loader, fold, checkpoint):
    """Main training function"""
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
        "val_loss": [],
        "train_bac": [],
        "val_bac": [],
    }
    best_bac = 0.0
    starting_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Load state_dict from checkpoint if desired
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch = checkpoint["epoch"]
        history = checkpoint["history"]

    for epoch in tqdm(range(starting_epoch, args.epochs)):
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
            "balanced_accuracy": train_metrics_computed['balanced_accuracy'].item(),
            "precision": train_metrics_computed['precision'].item(),
            "recall": train_metrics_computed['recall'].item(),
            "cohen_kappa": train_metrics_computed['cohen_kappa'].item(),
        }

        # Validation Metrics
        val_metrics = validate(model, val_loader, criterion)

        # Logging metrics for plotting
        history["train_loss"].append(train_loss/len(train_loader))
        history["val_loss"].append(val_metrics["loss"])
        history["train_bac"].append(train_metrics["balanced_accuracy"])
        history["val_bac"].append(val_metrics["balanced_accuracy"])


        # Printing metrics every 5 epochs and saving checkpoint
        if (epoch+1) % 5 == 0:
            print_metrics(train_metrics, val_metrics, epoch, fold)
            checkpoint = {
                'epoch': epoch,
                'fold': fold,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, args.checkpoint_dir)

        # Saving model based on balanced accuracy score
        if val_metrics["balanced_accuracy"] > best_bac:
            best_bac = val_metrics["balanced_accuracy"]
            checkpoint = {
                'epoch': epoch,
                'fold': fold,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, f"{args.model_dir}/best_model_f_{fold+1}.pth")

    return history

def validate(model, val_loader, criterion):
    """Main validation function

    returns:
        metrics: collected validation metrics
        """
    # Initialize validation metrics using MetricCollection
    val_metrics_tracker = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=args.num_classes),
        'balanced_accuracy': MulticlassAccuracy(num_classes=args.num_classes, average='macro'),
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
            "balanced_accuracy": val_metrics_computed['balanced_accuracy'].item(),
            "precision": val_metrics_computed['precision'].item(),
            "recall": val_metrics_computed['recall'].item(),
            "macro_f1": val_metrics_computed['f1_macro'].item(),
            "cohen_kappa": val_metrics_computed['cohen_kappa'].item(),
            "roc_auc_macro": roc_auc.compute().item(),
            "per_label_f1": f1_per_class.compute().cpu().numpy(),
            "per_label_precision": precision_per_class.compute().cpu().numpy(),
            "per_label_recall": recall_per_class.compute().cpu().numpy(),
            "confusion_matrix": val_metrics_computed['confusion_matrix'],
        }

        return metrics