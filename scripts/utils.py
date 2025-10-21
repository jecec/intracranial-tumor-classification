import cv2
import numpy as np
from pathlib import Path

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
    print(f"Macro Precision: {val['per_label_f1']}")
    print(f"Roc-AUC-Macro: {val['roc_auc_macro']:.4f}")
    print(f"Roc-AUC: {val['roc_auc']}")

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
        Path("output/models"),
        Path("CSVs"),
        Path("data"),
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

# TODO: Create helper function for plotting and visualization