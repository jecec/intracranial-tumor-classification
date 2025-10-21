import argparse
import torch
def get_args():
    parser = argparse.ArgumentParser('Model Training Arguments')

    # File paths
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('-data', type=str, default='data/brisc2025/classification_task')
    data_group.add_argument('-csv_dir', type=str, default='CSVs')
    data_group.add_argument('-output_dir', type=str, default='output')
    data_group.add_argument('-model_dir', type=str, default='output/models')
    data_group.add_argument('-visual_dir', type=str, default='output/visuals')

    # Training
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('-batch_size', type=int, default=32, choices=[16, 24, 32])
    train_group.add_argument('-num_workers', type=int, default=8, choices=[4, 6, 8])
    train_group.add_argument('-pre_fetch', type=int, default=4, choices=[1, 2, 4])
    train_group.add_argument('-epochs', type=int, default=5, choices=[25, 50, 100])
    train_group.add_argument('-lr', type=float, default=1e-3, choices=[1e-3, 1e-4, 1e-5])
    train_group.add_argument('-folds', type=int, default=5)

    # Backbone, seed and device
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('-backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    misc_group.add_argument('-seed', type=int, default=28)
    misc_group.add_argument('-device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()
    return args

