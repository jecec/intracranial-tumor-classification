import cv2
import numpy as np

def metrics(train, val, epoch, fold):
    """Print metrics of training and validation"""
    print(f"-- Fold: {fold+1}, Epoch: {epoch+1} --")
    print(f"Training loss: {train['loss']:.4f}")
    print(f"Validation loss: {val['loss']:.4f}")

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

# TODO: Create helper function for plotting and visualization