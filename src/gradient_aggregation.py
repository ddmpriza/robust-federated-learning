
import torch

# Compute the sample-size-weighted mean of client gradients.
# gradients: Tensor [num_clients, num_parameters]
# client_sizes: Tensor [num_clients]
def gradient_mean(gradients, client_sizes):
    if gradients.ndim != 2:
        raise ValueError("Expected a 2D tensor of client gradients.")

    if client_sizes.ndim != 1 or len(client_sizes) != len(gradients):
        raise ValueError("Client sizes must match the number of gradients.")

    if torch.any(client_sizes <= 0):
        raise ValueError("Client sizes must be positive.")

    weights = client_sizes.to(device=gradients.device, dtype=gradients.dtype)

    # Normalize client sizes to sum to 1
    weights = weights / weights.sum()

    # Compute the weighted mean of the gradients
    aggregated_gradient = ( gradients * weights.unsqueeze(1)).sum(dim=0)  # unsqueeze: Add a dimension to weights to allow broadcasting during multiplication
                                                                          # sum(dim=0): Sum the weighted gradients across all clients to get the final aggregated gradient 
                                                                          # dim=0 indicates summation along the first dimension (clients)

    return aggregated_gradient
