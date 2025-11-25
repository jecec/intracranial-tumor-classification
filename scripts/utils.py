import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from args import get_args

args = get_args()


def print_metrics(train_metrics=None, eval_metrics=None,  epoch=None, fold=None):
    """Function for printing metrics in training, validation and evaluation steps

    Args:
        train_metrics (dict): dictionary of training metrics
        eval_metrics (dict): dictionary of either validation or evaluation metrics
        epoch (int): epoch number
        fold (int): fold number
    """
    print("\n", "-" * 60)

    if train_metrics:
        if fold is not None:
            print(f"-- Fold: {fold + 1}, Epoch: {epoch + 1} --")
        elif epoch is not None:
            print(f"-- Epoch: {epoch + 1} --")

        print("Training metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {train_metrics['precision']:.4f}")
        print(f"  Recall (macro): {train_metrics['recall']:.4f}")
        print(f"  Cohen's Kappa: {train_metrics['cohen_kappa']:.4f}")
    if eval_metrics:
        print(f"Evaluation metrics:")
        print(f"  Loss: {eval_metrics['loss']:.4f}")
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {eval_metrics['precision']:.4f}")
        print(f"  Recall (macro): {eval_metrics['recall']:.4f}")
        print(f"  F1 Score (macro): {eval_metrics['macro_f1']:.4f}")
        print(f"  Cohen's Kappa: {eval_metrics['cohen_kappa']:.4f}")
        print(f"  ROC-AUC (macro): {eval_metrics['roc_auc_macro']:.4f}")

        print(f"  Per-class F1: {[f'{x:.3f}' for x in eval_metrics['per_label_f1'].tolist()]}")
        print(f"  Per-class Precision: {[f'{x:.3f}' for x in eval_metrics['per_label_precision'].tolist()]}")
        print(f"  Per-class Recall: {[f'{x:.3f}' for x in eval_metrics['per_label_recall'].tolist()]}")

    print("-"*60)

def plot_training_metrics(history, cfmx=None, fold=None):
    """Function for saving plots of training and validation metrics

    Args:
        history (dict): Dictionary of training loss and balanced accuracy metrics
        fold (int, optional): Fold number
    Outputs:
        plots of loss and balanced accuracy metrics over epochs. separate plots for each fold if any
    """

    # 1. Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss over Epochs", fontsize=14)
    plt.grid(True, alpha=0.3)
    if fold is not None:
        plt.savefig(f"{args.visual_dir}/loss_fold_{fold + 1}.png", dpi=150, bbox_inches='tight')
    else:
        plt.savefig(f"{args.visual_dir}/loss_main.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nLoss plot saved to {args.visual_dir}")

    # 2. Balanced Accuracy plot
    # TODO: Fix inconsistent naming, even though the functionality is the same
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_recall"], label="Training Balanced Accuracy", linewidth=2)
    if "val_recall" in history:
        plt.plot(history["val_recall"], label="Validation Balanced Accuracy", linewidth=2)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Balanced Accuracy", fontsize=12)
    plt.title("Balanced Accuracy Score over Epochs", fontsize=14)
    plt.grid(True, alpha=0.3)
    if fold is not None:
        plt.savefig(Path(args.visual_dir, f"balanced_accuracy_fold_{fold + 1}.png"), dpi=150, bbox_inches='tight')
    else:
        plt.savefig(Path(args.visual_dir,"balanced_accuracy_main.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Balanced Accuracy plot saved to {args.visual_dir}")

    # 3. Confusion Matrix
    if cfmx is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cfmx, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["glioma", "meningioma", "no_tumor", "pituitary"],
                    yticklabels=["glioma", "meningioma", "no_tumor", "pituitary"])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'Confusion Matrix - Fold {fold+1}', fontsize=14)
        plt.tight_layout()
        plt.savefig(Path(args.visual_dir, f"confusion_matrix_fold_{fold+1}.png"), dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {args.visual_dir}\n")

def aggregate_fold_metrics(fold_metrics):
    """Function for aggregating metrics across folds

    Args:
        fold_metrics: List of metric dictionaries from each fold

    Returns:
        aggregated: Dictionary containing aggregated scalar and per-label metrics
    """
    aggregated = {}

    # Compute mean and std for scalar metrics
    scalar_metrics = ["loss", "accuracy", "precision",
                      "recall", "macro_f1", "cohen_kappa", "roc_auc_macro"]

    for metric in scalar_metrics:
        values = [fold[metric] for fold in fold_metrics]
        aggregated[f"{metric}_mean"] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)
        aggregated[f"{metric}_values"] = values

    # Aggregate per-class metrics across folds
    per_label_metrics = ["per_label_f1", "per_label_precision", "per_label_recall"]

    for metric in per_label_metrics:
        # Using numpy to aggregate through arrays of per-class metrics
        values = np.array([fold[metric] for fold in fold_metrics])
        aggregated[f"{metric}_mean"] = np.mean(values, axis=0)
        aggregated[f"{metric}_std"] = np.std(values, axis=0)
        aggregated[f"{metric}_values"] = values

    # Saving aggregated metrics to pkl
    save_metrics_pkl(aggregated, "aggregate_kfold")

    return aggregated

def print_aggregated_metrics(aggregated_metrics):
    """Function for printing validation metrics aggregated over folds"""
    # Print aggregated results
    print(f"\n{'=' * 50}")
    print("AGGREGATED VALIDATION SET RESULTS (ACROSS ALL FOLDS)")
    print(f"{'=' * 50}")
    print(f"Loss: {aggregated_metrics['loss_mean']:.4f} ± {aggregated_metrics['loss_std']:.4f}")
    print(f"Accuracy: {aggregated_metrics['accuracy_mean']:.4f} ± {aggregated_metrics['accuracy_std']:.4f}")
    print(f"Precision (macro): {aggregated_metrics['precision_mean']:.4f} ± {aggregated_metrics['precision_std']:.4f}")
    print(f"Recall (macro): {aggregated_metrics['recall_mean']:.4f} ± {aggregated_metrics['recall_std']:.4f}")
    print(f"F1 Score (macro): {aggregated_metrics['macro_f1_mean']:.4f} ± {aggregated_metrics['macro_f1_std']:.4f}")
    print(f"Cohen's Kappa: {aggregated_metrics['cohen_kappa_mean']:.4f} ± {aggregated_metrics['cohen_kappa_std']:.4f}")
    print(f"ROC-AUC (macro): {aggregated_metrics['roc_auc_macro_mean']:.4f} ± {aggregated_metrics['roc_auc_macro_std']:.4f}")

    print(f"\nPer-class F1: {[f'{x:.3f}' for x in aggregated_metrics['per_label_f1_mean'].tolist()]} ± {[f'{x:.3f}' for x in aggregated_metrics['per_label_f1_std'].tolist()]}")
    print(f"Per-class Precision: {[f'{x:.3f}' for x in aggregated_metrics['per_label_precision_mean'].tolist()]} ± {[f'{x:.3f}' for x in aggregated_metrics['per_label_precision_std'].tolist()]}")
    print(f"Per-class Recall: {[f'{x:.3f}' for x in aggregated_metrics['per_label_recall_mean'].tolist()]} ± {[f'{x:.3f}' for x in aggregated_metrics['per_label_recall_std'].tolist()]}")
    print("-" * 50 + "\n")

def resize_pad(img, target_size=512):
    """Function for resizing and adding padding to images that are not in the standard 512x512 shape"""
    # Original Dimensions
    h, w = img.shape
    # Scale of resize
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding for target size
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    img_padded = np.pad(img_resized, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return img_padded

def save_metrics_pkl(metrics, phase, fold=None):
    """Function for saving metrics into a pickle file

    Output:
        best_model_fold_x_metrics.pkl: Metrics by fold stored in pickle in the output folder
        final_model_metrics.pkl: Metrics of the final trained model
    """
    if phase == "validate_kfold":
        filepath = Path(args.metrics_dir, f'best_model_fold_{fold}_metrics.pkl')
    elif phase == "evaluate_final":
        filepath = Path(args.metrics_dir, 'final_model_metrics.pkl')
    elif phase == "aggregate_kfold":
        filepath = Path(args.metrics_dir, 'aggregated_kfold_metrics.pkl')
    else: return
    with open(filepath, 'wb') as file:
        pickle.dump(metrics, file)