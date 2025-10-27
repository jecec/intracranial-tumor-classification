import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import *

from args import get_args
from utils import save_metrics_pkl, save_metrics_json
from model import PreTrainedModel

args = get_args()
device = args.device

def evaluate(test_loader):
    """Main evaluation function"""
    criterion = nn.CrossEntropyLoss()
    fold_metrics = []

    for fold, path in enumerate(Path(args.model_dir).iterdir()):
        model = PreTrainedModel(args.backbone, pretrained=False).to(device)
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        model.eval()

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

        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        metrics = {
            "loss": test_loss / len(test_loader),
            "accuracy": accuracy_score(all_targets, all_preds),
            "balanced_accuracy": balanced_accuracy_score(all_targets, all_preds),
            "macro_f1": f1_score(all_targets, all_preds, average="macro"),
            "per_label_f1": f1_score(all_targets, all_preds, average=None),
            "roc_auc_macro": roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro"),
            "per_label_roc_auc": roc_auc_score(all_targets, all_probs, multi_class="ovr", average=None),
            "confusion_matrix": confusion_matrix(all_targets, all_preds)
        }

        save_metrics_pkl(metrics, fold)
        save_metrics_json(metrics, fold)
        fold_metrics.append(metrics)

    return fold_metrics


