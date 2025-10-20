import torch
import pandas as pd

from dataset import prepare_dataset, k_fold_cv
from args import get_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # Fetching arguments
    args = get_args()

    ### PREPARING DATASET ###
    # Storing dataset into train and test csv files
    prepare_dataset(args.data)
    # Creating stratified 5-fold splits
    k_fold_cv(f'{args.csv_dir}/train_metadata.csv')

if __name__ == '__main__':
    main()