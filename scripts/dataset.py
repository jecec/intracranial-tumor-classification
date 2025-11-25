import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2

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

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    data = pd.read_csv(path)

    # Check if all fold files already exist
    all_folds_exist = all(
        os.path.exists(f"{args.csv_dir}/fold_{fold}_train.csv") and
        os.path.exists(f"{args.csv_dir}/fold_{fold}_val.csv")
        for fold in range(args.folds)
    )

    if all_folds_exist:
        print(f"All {args.folds} fold files already exist in {args.csv_dir}. Skipping k-fold creation.")
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

    mri_3ch = np.stack([mri, mri, mri], axis=-1)

    return mri_3ch

class MRI_dataset(Dataset):
    """Class for initializing dataset."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # Label map for numerical encodings
        self._label_map = {
            'glioma': 0,
            'meningioma': 1,
            'no_tumor': 2,
            'pituitary': 3
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset["Path"].iloc[idx]
        img = read_mri(path)
        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        label_str = self.dataset["Label"].iloc[idx]
        label = self._label_map[label_str]
        label = torch.tensor(label, dtype=torch.long)

        ## Visualizing images after transformations
        #import matplotlib.pyplot as plt
        #plt.imshow(img[0], cmap='gray')
        #plt.show()

        return {
            'img': img,
            'label': label
        }

def transforms(phase):
    """Function for applying transformations to dataset.

    args:
        phase: Either 'train' or 'val' for respective phases.
                Only resizing and normalization are applied during validation and evaluation.
    """
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(224, (0.8, 1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    if phase == 'train':
        return train_transform
    elif phase == 'val':
        return val_transform