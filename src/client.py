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