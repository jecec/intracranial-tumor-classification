from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_args()

def train(model, train_loader, val_loader, fold):
    """Main training function"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0

        for batch in train_loader:
            # Load batch data
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)

            # Resetting the gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # TODO: Add more metrics
        # Training Metrics
        train_metrics = {
            "loss": train_loss / len(train_loader),}
        # TODO: Add more metrics
        # Validation Metrics
        val_metrics = {
            "loss": validate(model, val_loader, criterion),}
        # Printing metrics every 2 epochs
        if (epoch + 1) % 25 == 0:
            metrics(train_metrics, val_metrics, epoch, fold)

        # TODO: Add plotting for training loss, validation loss and other metrics

        # TODO: Add model checkpoints and saving
def validate(model, val_loader, criterion):
    """Main validation function

    returns:
        val_loss: validation loss
        """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        return val_loss