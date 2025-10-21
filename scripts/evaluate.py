import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from args import get_args
from model import PreTrainedModel

args = get_args()
device = args.device

def evaluate(test_loader):
    """Main evaluation function"""
    for path in Path(args.model_dir).iterdir():
        model = PreTrainedModel(args.backbone, pretrained=False).to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        all_preds, all_targets = [], []
        for batch in test_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # TODO: Add more metrics for evaluation

        cm = confusion_matrix(targets, preds)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()


