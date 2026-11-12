import torch
from torch.utils.data import DataLoader

# Train a model locally using one client's dataset
def train_client(model, dataset, epochs=1, batch_size=32, learning_rate=0.01):
    model.train()

    # Dataloader for batching the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = torch.nn.CrossEntropyLoss()         # loss function for multi-class classification

    optimizer = torch.optim.SGD(                    # local optimizer for updating model parameters
        model.parameters(),
        lr=learning_rate
    )

    
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()               # Reset gradients from the previous batch

            outputs = model(images)             # Forward pass

            loss = criterion(outputs, labels)   # Calculate the loss

            loss.backward()                     # Calculate gradients

            optimizer.step()                    # Update model parameters
            
            total_loss += loss.item()           # Add the loss of this batch

        average_loss = total_loss / len(dataloader)     # Average loss of all batches in this epoch

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}")

    return model

def compute_client_gradient(model, dataset, batch_size=32):
    """Compute the average gradient over a client's dataset."""

    if len(dataset) == 0:
        raise ValueError("Client dataset must not be empty.")

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    parameters = list(model.parameters())
    gradient_sums = [torch.zeros_like(p) for p in parameters]

    for images, labels in loader:
        images = images.to(parameters[0].device)
        labels = labels.to(parameters[0].device)

        model.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for total, param in zip(gradient_sums, parameters):
            total.add_(param.grad.detach())

    mean_gradients = [
        total / len(dataset)
        for total in gradient_sums
    ]

    return torch.cat([g.reshape(-1) for g in mean_gradients])