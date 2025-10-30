import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassCohenKappa
)

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
        state = torch.load(path, map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(state)
        model.eval()

        # Initialize metrics using MetricCollection
        test_metrics_tracker = MetricCollection({
            'accuracy': MulticlassAccuracy(num_classes=args.num_classes),
            'balanced_accuracy': MulticlassAccuracy(num_classes=args.num_classes, average='macro'),
            'precision': MulticlassPrecision(num_classes=args.num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=args.num_classes, average='macro'),
            'f1_macro': MulticlassF1Score(num_classes=args.num_classes, average='macro'),
            'cohen_kappa': MulticlassCohenKappa(num_classes=args.num_classes),
            'confusion_matrix': MulticlassConfusionMatrix(num_classes=args.num_classes),
        }).to(device)

        # Separate metrics that need different inputs
        roc_auc = MulticlassAUROC(num_classes=args.num_classes, average='macro').to(device)
        f1_per_class = MulticlassF1Score(num_classes=args.num_classes, average=None).to(device)
        precision_per_class = MulticlassPrecision(num_classes=args.num_classes, average=None).to(device)
        recall_per_class = MulticlassRecall(num_classes=args.num_classes, average=None).to(device)

        test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['img'].to(device)
                targets = batch['label'].to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Update metrics
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)

                # Update all prediction-based metrics at once
                test_metrics_tracker.update(preds, targets)

                # Update probability-based and per-class metrics separately
                roc_auc.update(probs, targets)
                f1_per_class.update(preds, targets)
                precision_per_class.update(preds, targets)
                recall_per_class.update(preds, targets)

        # Compute all metrics
        test_metrics_computed = test_metrics_tracker.compute()
        metrics = {
            "loss": test_loss / len(test_loader),
            "accuracy": test_metrics_computed['accuracy'].item(),
            "balanced_accuracy": test_metrics_computed['balanced_accuracy'].item(),
            "precision": test_metrics_computed['precision'].item(),
            "recall": test_metrics_computed['recall'].item(),
            "macro_f1": test_metrics_computed['f1_macro'].item(),
            "cohen_kappa": test_metrics_computed['cohen_kappa'].item(),
            "roc_auc_macro": roc_auc.compute().item(),
            "per_label_f1": f1_per_class.compute().cpu().numpy(),
            "per_label_precision": precision_per_class.compute().cpu().numpy(),
            "per_label_recall": recall_per_class.compute().cpu().numpy(),
            "confusion_matrix": test_metrics_computed['confusion_matrix'].cpu().numpy(),
        }

        save_metrics_pkl(metrics, fold)
        #save_metrics_json(metrics, fold)
        fold_metrics.append(metrics)

    return fold_metrics


