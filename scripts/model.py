from args import get_args
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights

args = get_args()

class PreTrainedModel(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=4, pretrained=True):
        super(PreTrainedModel, self).__init__()

        # Select weights based on the pretrained flag
        if backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.model = resnet34(weights=weights)
        elif backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)