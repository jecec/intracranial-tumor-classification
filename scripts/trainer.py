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

def train(model, train_loader, val_loader, fold):
    """Main training function"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0

        train_targets, train_preds, train_probs = [], [], []
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
            train_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())


        train_targets = np.concatenate(train_targets, axis=0)
        train_preds = np.concatenate(train_preds, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)

        # Training Metrics
        train_metrics = {
            "loss": train_loss / len(train_loader),
            "balanced_accuracy": balanced_accuracy_score(train_targets, train_preds),

        }
        # Validation Metrics
        val_metrics = validate(model, val_loader, criterion)
        if epoch % 5 == 0 & epoch != 0:
            print_metrics(train_metrics, val_metrics, epoch, fold)
        if val_metrics["balanced_accuracy"] > best_acc:
            best_acc = val_metrics["balanced_accuracy"]
            '''
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
            '''
            torch.save(model.state_dict(), f"{args.model_dir}/best_model_f_{fold+1}.pth")

        # TODO: Add plotting for training loss, validation loss and other metrics

        # TODO: Add model checkpoints
def validate(model, val_loader, criterion):
    """Main validation function

    returns:
        metrics: collected validation metrics
        """

    val_loss = 0
    val_targets, val_preds, val_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            model.eval()
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

            val_targets.append(targets.detach().cpu().numpy())
            val_preds.append(torch.argmax(outputs, dim= 1).detach().cpu().numpy())
            val_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

        val_targets = np.concatenate(val_targets, axis=0)
        val_preds = np.concatenate(val_preds, axis=0)
        val_probs = np.concatenate(val_probs, axis=0)

        metrics = {
            "loss": val_loss / len(val_loader),
            "balanced_accuracy": balanced_accuracy_score(val_targets, val_preds),
            "macro_f1": f1_score(val_targets, val_preds, average="macro"),
            "per_label_f1": f1_score(val_targets, val_preds, average=None),
            "roc_auc_macro": roc_auc_score(val_targets, val_probs, multi_class="ovr", average="macro"),
            "roc_auc": roc_auc_score(val_targets, val_probs, multi_class="ovr", average=None),


        }
        return metrics