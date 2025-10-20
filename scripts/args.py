import argparse

def get_args():
    parser = argparse.ArgumentParser('Model Training Arguments')
    parser.add_argument('-seed', type=int, default=28)
    parser.add_argument('-backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('-data', type=str, default='data/brisc2025/classification_task')
    parser.add_argument('-folds', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=64,
                        choices=[32, 64, 128, 256])
    parser.add_argument('-lr', type=float, default=1e-3,
                        choices=[1e-3, 1e-4, 1e-5])
    parser.add_argument('-epochs', type=int, default=1000,
                        choices=[500, 1000, 2000])
    parser.add_argument('-output_dir', type=str, default='output',)

    return parser.parse_args()

