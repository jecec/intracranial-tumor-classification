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

        # Looking at layers to freeze conv1 (bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        #for name, layer in self.model.named_children():
        #    print(name)

        # Freezing chosen model layers
        for name, layer in self.model.named_children():
            if name in ['conv1', 'bn1', 'layer1']:
                for param in layer.parameters():
                    param.requires_grad = False

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        # Add a custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x