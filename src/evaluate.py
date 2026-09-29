import torch
from torch.utils.data import DataLoader


def evaluate(model, dataset, batch_size=32):

    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)                             # Forward pass

            max_values, predicted = torch.max(outputs, 1)       # Select the class with the highest logit

            correct += (predicted == labels).sum().item()       # Count correct predictions

            total += labels.size(0)                             # Count total samples

    accuracy = correct / total

    return accuracy