import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from args import get_args

args = get_args()

def prepare_dataset(path):
    """Prepare a csv of mri data by extracting labels from directory names.

    Args:
        path: Path to root directory containing 'train' and 'test' subfolders

    Output:
        Saves the prepared metadata to the output folder in csv format.
        Creates train_metadata.csv and test_metadata.csv.
    """

    # Define splits and output paths
    splits = ['train', 'test']

    for split in splits:
        output_path = f"{args.output_dir}/{split}_metadata.csv"

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
                    image_path = os.path.join(split_path, label, image.name)
                    data.append({
                        'Name': image_path,
                        'Label': label
                    })

        metadata = pd.DataFrame(data)

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
        os.path.exists(f"{args.output_dir}/fold_{fold}_train.csv") and
        os.path.exists(f"{args.output_dir}/fold_{fold}_val.csv")
        for fold in range(args.folds)
    )

    if all_folds_exist:
        print(f"All {args.folds} fold files already exist in {args.output_dir}. Skipping k-fold creation.")
        return

    for fold, (train_index, val_index) in enumerate(skf.split(X=data['Name'], y=data['Label']), start=0):
        train_data = data.iloc[train_index]
        train_data.to_csv(f"{args.output_dir}/fold_{fold}_train.csv", index=False)
        val_data = data.iloc[val_index]
        val_data.to_csv(f"{args.output_dir}/fold_{fold}_val.csv", index=False)
        print(f"Fold {fold}: {len(train_data)} train, {len(val_data)} val samples")

