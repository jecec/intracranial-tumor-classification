import torch
import pandas as pd
from torch.utils.data import DataLoader
import os

from dataset import prepare_dataset, k_fold_cv, MRI_dataset
from args import get_args
from model import preTrainedModel
from trainer import train

def main():

    # Fetch arguments and get device
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### PREPARE CSVs ###
    # Storing dataset into train and test csv files
    prepare_dataset(args.data)
    # Creating stratified 5-fold splits
    k_fold_cv(f'{args.csv_dir}/train_metadata.csv')

    ### ITERATE AMONG THE FOLDS ###
    for fold in range(args.n_folds):
        print(f"Training on fold {fold + 1} out of {args.n_folds}")

        # 1. Load datasets
        train_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_train.csv'))
        val_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_val.csv'))

        # 2. Prepare datasets
        train_dataset = MRI_dataset(train_set)
        val_dataset = MRI_dataset(val_set)

        # 3. Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                                  prefetch_factor=args.pre_fetch, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                                prefetch_factor=args.pre_fetch, persistent_workers=True)

        # 4. Initialize the model
        model = preTrainedModel(args.backbone).to(device)

        # 5. Training the model
        train(model, train_loader, val_loader, fold)

        # TODO: Add model evaluation


if __name__ == '__main__':
    main()