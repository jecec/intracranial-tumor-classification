import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from sklearn.metrics import *

from args import get_args
from model import PreTrainedModel

args = get_args()
device = args.device

def evaluate(test_loader):
    """Main evaluation function"""
    criterion = nn.CrossEntropyLoss()
    fold_metrics = []
    for path in Path(args.model_dir).iterdir():
        model = PreTrainedModel(args.backbone, pretrained=False).to(device)
        model.load_state_dict(torch.load(path, weights_only=True))

        test_loss = 0
        all_targets, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['img'].to(device)
                targets = batch['label'].to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()

                all_targets.append(targets.detach().cpu().numpy())
                all_preds.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())
                all_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

        targets = np.concatenate(all_targets, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        probs = np.concatenate(all_probs, axis=0)

        metrics = {
            "loss": test_loss / len(test_loader),
            "accuracy": accuracy_score(targets, preds),
            "balanced_accuracy": balanced_accuracy_score(targets, preds),
            "macro_f1": f1_score(targets, preds, average="macro"),
            "per_label_f1": f1_score(targets, preds, average=None),
            "roc_auc_macro": roc_auc_score(targets, probs, multi_class="ovr", average="macro"),
            "per_label_roc_auc": roc_auc_score(targets, probs, multi_class="ovr", average=None),
            "confusion_matrix": confusion_matrix(targets, preds)
        }

        fold_metrics.append(metrics)

    return fold_metrics


