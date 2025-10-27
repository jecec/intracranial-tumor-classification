import torch
import pandas as pd
from torch.utils.data import DataLoader
import os

from dataset import prepare_dataset, k_fold_cv, MRI_dataset, transforms
from args import get_args
from model import PreTrainedModel
from scripts.utils import aggregate_metrics
from trainer import train
from evaluate import evaluate
from utils import setup_project_dirs, plot_training_metrics, evaluation_metrics

def main():
    # TODO: Add skip training for evaluation of the model only
    # Fetch arguments and get device
    args = get_args()
    device = args.device
    checkpoint = None
    starting_fold = 0

    # Prepare folder structure
    setup_project_dirs()

    ### PREPARE CSVs ###
    # Storing dataset into train and test csv files
    prepare_dataset(args.data)
    # Creating stratified 5-fold splits
    k_fold_cv(f'{args.csv_dir}/train_metadata.csv')

    # Loading checkpoints
    if args.resume:
        checkpoint = torch.load(args.checkpoint_dir, map_location="cpu")
        starting_fold = checkpoint["fold"]
        resume_training = True
        print(f"\nResuming training from fold {starting_fold+1} and epoch {checkpoint['epoch']+1}")
    else:
        resume_training = False

    ### ITERATE AMONG THE FOLDS AND TRAIN THE MODEL ###
    for fold in range(starting_fold, args.folds):
        print(f"\nTraining on fold {fold + 1} out of {args.folds}")

        # 1. Load datasets
        train_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_train.csv'))
        val_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_val.csv'))


        # 2. Prepare datasets
        train_dataset = MRI_dataset(train_set, transform = transforms('train'))
        val_dataset = MRI_dataset(val_set, transform = transforms('val'))

        # 3. Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                                  prefetch_factor=args.pre_fetch, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                                prefetch_factor=args.pre_fetch, persistent_workers=True)

        # 4. Initialize model with checkpoint weights or pretrained weights
        if resume_training:
            model = PreTrainedModel(args.backbone, pretrained=False)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
        else:
            model = PreTrainedModel(args.backbone, pretrained=True).to(device)

        # 5. Training the model
        history= train(model, train_loader, val_loader, fold, checkpoint)
        # 6. Visualize metrics
        plot_training_metrics(history, fold)

        # Setting variables so only the initial loop is loaded from checkpoint
        resume_training = False
        checkpoint = None

    ## MODEL EVALUATION ##
    test_set = pd.read_csv(f'{args.csv_dir}/test_metadata.csv')
    test_dataset = MRI_dataset(test_set, transform = transforms('val'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    all_metrics = evaluate(test_loader)
    aggregated = aggregate_metrics(all_metrics)

    # Printing evaluation metrics and saving plots
    evaluation_metrics(aggregated, all_metrics)

if __name__ == '__main__':
    main()