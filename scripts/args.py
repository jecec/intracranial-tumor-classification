import argparse
import torch
def get_args():
    # TODO: Split arguments to config.yaml and argparser for usability
    parser = argparse.ArgumentParser('Model Training Arguments')

    # File paths
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('-data', type=str, default='data/brisc2025/classification_task')
    data_group.add_argument('-csv_dir', type=str, default='data/CSVs')
    data_group.add_argument('-output_dir', type=str, default='output')
    data_group.add_argument('-model_dir', type=str, default='output/models')
    data_group.add_argument('-visual_dir', type=str, default='output/visuals')
    data_group.add_argument('-checkpoint_dir', type=str, default='output/checkpoints')
    data_group.add_argument('-metrics_dir', type=str, default='output/metrics')

    # Training hyperparameters
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('-batch_size', type=int, default=32, choices=[16, 24, 32])
    train_group.add_argument('-num_workers', type=int, default=8, choices=[4, 6, 8])
    train_group.add_argument('-pre_fetch', type=int, default=4, choices=[1, 2, 4])
    train_group.add_argument('-epochs', type=int, default=8, choices=[5, 10, 15])
    train_group.add_argument('-lr', type=float, default=1e-4, choices=[1e-3, 1e-4, 1e-5])
    train_group.add_argument('-folds', type=int, default=5)


    # Miscellaneous options such as resume for resuming training from saved checkpoints
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('-backbone', type=str, default='resnet34', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    misc_group.add_argument('-seed', type=int, default=28)
    misc_group.add_argument('-device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    misc_group.add_argument('-resume', type=bool, default=True, choices=[True, False])
    misc_group.add_argument('-num_classes', type=int, default=4)
    misc_group.add_argument('-print_rate', type=int, default=4)
    misc_group.add_argument('-train', type=bool, default=False)
    misc_group.add_argument('-evaluate', type=bool, default=True)

    args = parser.parse_args()
    return args

