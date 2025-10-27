import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tabulate import tabulate
import json
import copy
import os
import pickle

from args import get_args
args = get_args()

def print_metrics(train, val, epoch, fold):
    """Print metrics of training and validation"""
    print(f"\n-- Fold: {fold+1}, Epoch: {epoch+1} --")
    print("Training metrics:")
    print(f"Loss: {train['loss']:.4f}")
    print(f"Balanced Accuracy: {train['balanced_accuracy']:.4f}")

    print(f"\nValidation metrics:")
    print(f"Loss: {val['loss']:.4f}")
    print(f"Balanced Accuracy: {val['balanced_accuracy']:.4f}")
    print(f"Macro F1 Score: {val['macro_f1']:.4f}")
    print(f"F1 Precision per Label: {val['per_label_f1']}")
    print(f"Roc-AUC-Macro: {val['roc_auc_macro']:.4f}")
    print(f"Roc-AUC per Label: {val['per_label_roc_auc']}")
    print("------------------------")

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

def show_img(img):
    """Helper function for visualizing images"""
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def setup_project_dirs():
    """Helper function for setting up required project dirs"""
    folders = [
        Path(args.model_dir),
        Path(args.csv_dir),
        Path("data"),
        Path(args.visual_dir)
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

def plot_training_metrics(history, fold):
    """Function saving plots of training and validation metrics"""

    # 1. Loss of train and val
    plt.figure()
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.savefig(f"{args.visual_dir}/loss_fold_{fold}.png")
    plt.close()

    # 2. Balanced Accuracy of train and val
    plt.figure()
    plt.plot(history["train_bac"], label="Training BAS")
    plt.plot(history["val_bac"], label="Validation BAS")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Balanced Accuracy Score over Epochs")
    plt.savefig(f"{args.visual_dir}/BAS_fold_{fold}.png")
    plt.close()

def evaluation_metrics(aggregated, all_metrics):
    """Function for plotting and saving evaluation metrics"""

    # Print mean and std of aggregated scalar metrics
    headers = ["Metric", "Mean", "Std"]
    data = []
    for key, value in aggregated["scalar"].items():
        mean = np.round(value["mean"], 3)
        std = np.round(value["std"], 3)
        data.append([str(key), mean, std])
    print("\n-- Aggregated evaluation metrics --")
    print(tabulate(data, headers=headers))

    # Print mean and std of per class metrics by class
    labels = ["glioma", "meningioma", "no_tumor", "pituitary"]
    headers = ["Class", "F1 (mean)", "F1 (std)", "ROC AUC (mean)", "ROC AUC (std)"]
    metrics = aggregated["per_label"]

    per_label_data = []
    for i, label in enumerate(labels):
        per_label_data.append([
            label,
            np.round(metrics["per_label_f1"]["mean"][i], 3),
            np.round(metrics["per_label_f1"]["std"][i], 3),
            np.round(metrics["per_label_roc_auc"]["mean"][i], 3),
            np.round(metrics["per_label_roc_auc"]["std"][i], 3)
        ])
    print("\n-- Aggregated evaluation metrics per label --")
    print(tabulate(per_label_data, headers=headers))

    # Save confusion matrices for each fold
    for fold in range(args.folds):
        vis = ConfusionMatrixDisplay(all_metrics[fold]["confusion_matrix"])
        vis.plot()
        plt.savefig(f"{args.visual_dir}/confusion_matrix_f_{fold}.png")
        plt.close()

def aggregate_metrics(metrics):
    """Function for aggregating metrics across folds"""
    aggregated = {}
    scalar_metric = {}
    array_metric = {}

    # Compute mean and std for scalar metrics
    scalar_metrics = ["loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc_macro"]

    for metric in scalar_metrics:
        values = [fold[metric] for fold in metrics]
        scalar_metric[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values
        }

    # Aggregate per-class metrics across folds
    per_label_metrics = ["per_label_f1", "per_label_roc_auc"]

    for metric in per_label_metrics:
        # Using numpy to aggregate through arrays of per-class metrics
        values = np.array([fold[metric] for fold in metrics])

        array_metric[metric] =  {
            "metric": metric,
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
            "values": values
        }
    aggregated["scalar"] = scalar_metric
    aggregated["per_label"] = array_metric
    return aggregated

def save_metrics_pkl(metrics, fold):
    """Function for saving metrics into a json file

    Output:
        metrics.json: Metrics by fold stored in json in the output folder
    """
    filepath = os.path.join(args.output_dir, f'metrics_fold_{fold}.pkl')
    with open(filepath, 'wb') as file:
        pickle.dump(metrics, file)

def save_metrics_json(metrics_to_save, fold):
    """Function for saving metrics into a json file

    Output:
        metrics.json: Metrics by fold stored in json in the output folder
    """
    metrics = copy.deepcopy(metrics_to_save)
    if fold == 1:
        os.remove(f'{args.output_dir}/metrics.json')
    # Process metrics
    processed_metrics = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        if type(value) == np.ndarray:
            processed_metrics[key] = value.tolist()
        else:
            processed_metrics[key] = value

    # Read existing data if file exists
    filepath = f'{args.output_dir}/metrics.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            all_metrics = json.load(file)
    else:
        all_metrics = {}

    # Add the new fold
    all_metrics[f'fold_{fold}'] = processed_metrics

    # Write back to file
    with open(filepath, 'w') as file:
        json.dump(all_metrics, file, indent=4)