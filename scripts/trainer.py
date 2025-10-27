from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import *
from tqdm import tqdm
import numpy as np

from utils import print_metrics

args = get_args()
device = args.device

def train(model, train_loader, val_loader, fold, checkpoint):
    """Main training function"""
    # TODO: Implement early stopping
    # TODO: Add more hyperparameters and training options here, such as batch normalization, learning rate scheduler etc.

    best_bac = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_bac": [],
        "val_bac": [],
    }
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
        train_targets, train_preds = [], []
        for batch in train_loader:
            # Load batch data
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)

            # Resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_targets.append(targets.detach().cpu().numpy())
            train_preds.append(torch.argmax(outputs, dim= 1).detach().cpu().numpy())

        train_targets = np.concatenate(train_targets, axis=0)
        train_preds = np.concatenate(train_preds, axis=0)

        # Training Metrics
        train_metrics = {
            "loss": train_loss / len(train_loader),
            "accuracy": accuracy_score(train_targets, train_preds),
            "balanced_accuracy": balanced_accuracy_score(train_targets, train_preds),

        }
        # Validation Metrics
        val_metrics = validate(model, val_loader, criterion)

        # Logging metrics for plotting
        history["train_loss"].append(train_loss)
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

    val_loss = 0
    val_targets, val_preds, val_probs = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

            val_targets.append(targets.detach().cpu().numpy())
            val_preds.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            val_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

        val_targets = np.concatenate(val_targets, axis=0)
        val_preds = np.concatenate(val_preds, axis=0)
        val_probs = np.concatenate(val_probs, axis=0)

        # TODO: Look deeper into metrics and collect only ones which are useful in this task
        metrics = {
            "loss": val_loss / len(val_loader),
            "accuracy": accuracy_score(val_targets, val_preds),
            "balanced_accuracy": balanced_accuracy_score(val_targets, val_preds),
            "macro_f1": f1_score(val_targets, val_preds, average="macro"),
            "per_label_f1": f1_score(val_targets, val_preds, average=None),
            "roc_auc_macro": roc_auc_score(val_targets, val_probs, multi_class="ovr", average="macro"),
            "per_label_roc_auc": roc_auc_score(val_targets, val_probs, multi_class="ovr", average=None),
            "confusion_matrix": confusion_matrix(val_targets, val_preds),
        }

        return metrics