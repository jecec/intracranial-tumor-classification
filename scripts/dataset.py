import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch

from args import get_args
from utils import resize_pad

args = get_args()

def prepare_dataset(path):
    """Prepare a csv of MRI data by extracting labels from directory names.

    Args:
        path: Path to root directory containing 'train' and 'test' subfolders

    Output:
        Saves the prepared metadata to the output folder in csv format.
        Creates train_metadata.csv and test_metadata.csv.
    """

    # Define splits and output paths
    splits = ['train', 'test']

    for split in splits:
        output_path = f"{args.csv_dir}/{split}_metadata.csv"

        # Check if metadata file already exists
        if os.path.exists(output_path):
            print(f"Metadata file already exists at {output_path}. Skipping preparation.")
            continue

        split_path = Path(path) / split

        data = []
        for folder in split_path.iterdir():
            if folder.is_dir():
                label = folder.name
                for image in folder.glob('*.jpg'):
                    data.append([str(image), label])
        # Create DataFrame
        metadata = pd.DataFrame(data, columns=['Path', 'Label'])

        # Save DataFrame to csv
        metadata.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path} with {len(metadata)} images")

def k_fold_cv(path):
    """Create stratified k-fold splits from the metadata CSV

    Output:
        Saves the k fold splits into enumerated training and validation folds.
        Creates {fold}_{index}_train.csv and {fold}_{index}_train.csv.
    """

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    data = pd.read_csv(path)

    # Check if all fold files already exist
    all_n_folds_exist = all(
        os.path.exists(f"{args.csv_dir}/fold_{fold}_train.csv") and
        os.path.exists(f"{args.csv_dir}/fold_{fold}_val.csv")
        for fold in range(args.n_folds)
    )

    if all_n_folds_exist:
        print(f"All {args.n_folds} fold files already exist in {args.csv_dir}. Skipping k-fold creation.")
        return

    # If folds don't exist, create them
    for fold, (train_index, val_index) in enumerate(skf.split(X=data['Path'], y=data['Label']), start=0):
        train_data = data.iloc[train_index]
        train_data.to_csv(f"{args.csv_dir}/fold_{fold}_train.csv", index=False)
        val_data = data.iloc[val_index]
        val_data.to_csv(f"{args.csv_dir}/fold_{fold}_val.csv", index=False)
        print(f"Fold {fold}: {len(train_data)} train, {len(val_data)} val samples")

def read_mri(path):
    """Read MRI images and resize non-uniform images with padding.
    After resizing these, resize all images to 224x224 as required by ResNet

    returns:
        mri_3ch: Greyscale images of MRI with suitable dimensions for ResNet.
    """
    mri = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Add padding and resizing images if they are not 512x512 for uniformity
    if mri.shape != (512, 512):
        mri = resize_pad(mri)
    # Resizing image to 224x224 required by ResNet
    mri = cv2.resize(mri, (224, 224))
    # Normalizing the images
    mri = mri.astype(np.float32) / 255.0
    # Converting greyscale images to a suitable shape for ResNet
    mri_3ch = np.zeros((3, mri.shape[0], mri.shape[1]), dtype=mri.dtype)
    mri_3ch[0] = mri
    mri_3ch[1] = mri
    mri_3ch[2] = mri

    return mri_3ch

class MRI_dataset(Dataset):
    """Class for initializing dataset."""
    def __init__(self, dataset):
        self.dataset = dataset

        # Label map for numerical encodings
        self.label_map = {
            'pituitary': 0,
            'meningioma': 1,
            'glioma': 2,
            'no_tumor': 3
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_mri(self.dataset["Path"].iloc[idx])
        label_str = self.dataset["Label"].iloc[idx]

        # Convert string labels to integer
        label = self.label_map[label_str]

        # Convert to tensors
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(label, dtype=torch.long)

        res = {
            'img': img_tensor,
            'label': label_tensor
        }
        return res

    # TODO: Add pre-processing and transformations if not enough data