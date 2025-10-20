from args import get_args
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights

args = get_args()

class preTrainedModel(nn.Module):
    def __init__(self, backbone=args.backbone, num_classes=4):
        super(preTrainedModel, self).__init__()
        if backbone == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == 'resnet34':
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif backbone == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # TODO: Experiment with layer freezing
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)