import torch
import torch.nn as nn
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from args import get_args
from model import PreTrainedModel

args = get_args()
device = args.device


def load_fold_models(model_paths):
    """Load all k-fold models

    Args:
        model_paths: List of paths to trained model weights

    Returns:
        models: List of loaded models in eval mode
    """
    models = []
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = PreTrainedModel(args.backbone, pretrained=False).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"Loaded model: {model_path}")

    return models


def evaluate_ensemble(test_loader, model_paths):
    """Evaluate ensemble of k-fold models on test set

    Args:
        test_loader: DataLoader for test set
        model_paths: List of paths to trained model weights (one per fold)

    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    # TODO: Add std tracking for predictions and flag samples with high variance
    criterion = nn.CrossEntropyLoss()

    print(f"\nEvaluating ensemble of {len(model_paths)} models")

    # Load all models
    models = load_fold_models(model_paths)

    # Initialize metrics
    test_metrics_tracker = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=args.num_classes),
        'balanced_accuracy': MulticlassAccuracy(num_classes=args.num_classes, average='macro'),
        'precision': MulticlassPrecision(num_classes=args.num_classes, average='macro'),
        'recall': MulticlassRecall(num_classes=args.num_classes, average='macro'),
        'f1_macro': MulticlassF1Score(num_classes=args.num_classes, average='macro'),
        'cohen_kappa': MulticlassCohenKappa(num_classes=args.num_classes),
        'confusion_matrix': MulticlassConfusionMatrix(num_classes=args.num_classes),
    }).to(device)

    roc_auc = MulticlassAUROC(num_classes=args.num_classes, average='macro').to(device)
    f1_per_class = MulticlassF1Score(num_classes=args.num_classes, average=None).to(device)
    precision_per_class = MulticlassPrecision(num_classes=args.num_classes, average=None).to(device)
    recall_per_class = MulticlassRecall(num_classes=args.num_classes, average=None).to(device)

    test_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Ensemble"):
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)

            # Collect predictions from all models
            all_logits = []
            all_probs = []

            for model in models:
                outputs = model(inputs)
                all_logits.append(outputs)
                all_probs.append(torch.softmax(outputs, dim=1))

            # Stack predictions
            all_logits = torch.stack(all_logits)
            all_probs = torch.stack(all_probs)

            # Ensemble prediction: average probabilities (soft voting)
            ensemble_probs = all_probs.mean(dim=0)
            ensemble_preds = ensemble_probs.argmax(dim=1)

            # Calculate loss using ensemble
            ensemble_logits = all_logits.mean(dim=0)
            loss = criterion(ensemble_logits, targets)
            test_loss += loss.item()

            # Update metrics with ensemble predictions
            test_metrics_tracker.update(ensemble_preds, targets)
            roc_auc.update(ensemble_probs, targets)
            f1_per_class.update(ensemble_preds, targets)
            precision_per_class.update(ensemble_preds, targets)
            recall_per_class.update(ensemble_preds, targets)

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
        "num_models": len(models),
    }

    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                xticklabels = ["glioma", "meningioma", "no_tumor", "pituitary"],
                yticklabels = ["glioma", "meningioma", "no_tumor", "pituitary"]
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{args.visual_dir}/confusion_matrix_ensemble.png", dpi=150)
    plt.close()

    return metrics