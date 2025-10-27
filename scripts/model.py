from args import get_args
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

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
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            self.model = resnet101(weights=weights)

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        # Add a custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x