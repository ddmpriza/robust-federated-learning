
import torch
from torch import nn
from torch.utils.data import DataLoader

# Compute the mean gradient over a client's dataset. 
def compute_client_gradient(model, dataset, batch_size=32):
    if len(dataset) == 0:
        raise ValueError("Client dataset must not be empty.")

    model.eval()                                                        # Set the model to evaluation mode to disable dropout and batch normalization layers
    criterion = nn.CrossEntropyLoss(reduction="sum")                    # Use sum reduction to accumulate the loss over the entire dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Create a DataLoader to iterate over the dataset in batches without shuffling

    parameters = list(model.parameters())                               # Returns a list of all the parameters in the model, organized by layer and type (weights, biases, etc.)
    gradient_sums = [torch.zeros_like(p) for p in parameters]           # Initialize a list of tensors to accumulate the gradients for each parameter, with the same shape as the corresponding parameter

    for images, labels in loader:                                       # Iterate over the DataLoader to get batches of images and labels
        images = images.to(parameters[0].device)
        labels = labels.to(parameters[0].device)

        model.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for total, param in zip(gradient_sums, parameters):     # Iterate over the accumulated gradients and the model parameters simultaneously
            total.add_(param.grad.detach())

    # Compute the mean gradient for each parameter by dividing the accumulated gradients by the total number of samples in the dataset
    mean_gradients = [total / len(dataset) for total in gradient_sums]

    # Concatenate all the mean gradients into a single 1D tensor by flattening each gradient tensor and concatenating them along the first dimension
    return torch.cat([g.reshape(-1) for g in mean_gradients])       