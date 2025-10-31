import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import prepare_dataset, k_fold_cv, MRI_dataset, transforms
from args import get_args
from model import PreTrainedModel
from scripts.utils import evaluation_metrics, aggregate_fold_metrics, print_aggregated_metrics
from utils import plot_training_metrics
from train_cv import train_cv
from train_final import train_final
from evaluate import evaluate


def main():
    # Fetch arguments and get device
    args = get_args()
    device = args.device
    checkpoint = None

    ### PREPARE CSVs ###
    # Storing dataset into train and test csv files
    prepare_dataset(args.data)

    # Only create K-fold splits if using cross-validation
    if args.train_cv:
        k_fold_cv(f'{args.csv_dir}/train_metadata.csv')

    ### K-FOLD CROSS-VALIDATION TRAINING ###
    if args.train_cv:
        fold_metrics = []
        starting_fold = 0
        resume_training = False

        if args.resume:
            try:
                checkpoint = torch.load(Path(args.checkpoint_dir, "checkpoint_cv.pth"), weights_only=False,
                                        map_location="cpu")
                starting_fold = checkpoint["fold"]
                fold_metrics = checkpoint["all_fold_metrics"]
                print(f"\nLoaded checkpoint {Path(args.checkpoint_dir, 'checkpoint_cv.pth')}")
                print(f"Resuming training from Fold {starting_fold + 1} and Epoch {checkpoint['epoch'] + 1}")
                resume_training = True
            except FileNotFoundError:
                print("\nCheckpoint not found, training from scratch")

        print("\n" + "=" * 60)
        print("STARTING K-FOLD CROSS-VALIDATION TRAINING")
        print("=" * 60)

        for fold in range(starting_fold, args.folds):
            print(f"\n-- Training on Fold {fold + 1} --")

            # 1. Load datasets
            train_set = pd.read_csv(Path(args.csv_dir, f'fold_{fold}_train.csv'))
            val_set = pd.read_csv(Path(args.csv_dir, f'fold_{fold}_val.csv'))

            # 2. Prepare datasets
            train_dataset = MRI_dataset(train_set, transform=transforms('train'))
            val_dataset = MRI_dataset(val_set, transform=transforms('val'))

            # 3. Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=args.pre_fetch,
                persistent_workers=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=args.pre_fetch,
                persistent_workers=True
            )

            # 4. Initialize model with checkpoint weights or pretrained weights
            if resume_training and checkpoint is not None:
                model = PreTrainedModel(args.backbone, pretrained=False)
                model.load_state_dict(checkpoint['state_dict'])
                model = model.to(device)
            else:
                model = PreTrainedModel(args.backbone, pretrained=True).to(device)

            # 5. Train the model
            # Pass all_fold_metrics to ensure checkpoint saves complete history
            history, val_metrics = train_cv(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                fold=fold,
                checkpoint=checkpoint if resume_training else None,
                all_fold_metrics=fold_metrics
            )

            # Append current fold's best validation metrics
            fold_metrics.append(val_metrics)

            # 6. Visualize metrics
            plot_training_metrics(history = history, cfmx = val_metrics["confusion_matrix"], fold=fold)

            # Reset checkpoint for later folds
            resume_training = False
            checkpoint = None

        # Display aggregated validation metrics
        aggregated_metrics = aggregate_fold_metrics(fold_metrics)
        print_aggregated_metrics(aggregated_metrics)

    ### SINGLE MODEL TRAINING ###
    if args.train_main:
        checkpoint = None
        resume_training = False
        if args.resume:
            try:
                checkpoint = torch.load(Path(args.checkpoint_dir, "checkpoint_main.pth"), weights_only=False,
                                        map_location="cpu")
                print(f"\nLoaded checkpoint {Path(args.checkpoint_dir, 'checkpoint_main.pth')}")
                print(f"Resuming training from Epoch {checkpoint['epoch'] + 1}")
                resume_training = True
            except FileNotFoundError:
                print("\nCheckpoint not found, training from scratch")

        print("\n" + "=" * 60)
        print("STARTING SINGLE MODEL TRAINING")
        print("=" * 60)

        # Load full training set
        train_set = pd.read_csv(Path(args.csv_dir, 'train_metadata.csv'))
        train_dataset = MRI_dataset(train_set, transform=transforms('train'))

        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=args.pre_fetch,
            persistent_workers=True
        )

        # Initialize model with checkpoint weights or pretrained weights
        if resume_training:
            model = PreTrainedModel(args.backbone, pretrained=False)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
        else:
            model = PreTrainedModel(args.backbone, pretrained=True).to(device)

        # Train the model
        history = train_final(model=model, train_loader=train_loader, checkpoint=checkpoint)

        # Visualize metrics
        plot_training_metrics(history)

    ### MODEL EVALUATION ###
    if args.evaluate:
        print("\n" + "=" * 60)
        print("STARTING MODEL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load test set
        test_set = pd.read_csv(f'{args.csv_dir}/test_metadata.csv')
        test_dataset = MRI_dataset(test_set, transform=transforms('val'))
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        test_metrics = evaluate(test_loader, Path(args.model_dir, 'final_model.pth'))
        evaluation_metrics(test_metrics)

if __name__ == '__main__':
    main()