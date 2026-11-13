
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

# Apply an aggregated gradient to update model parameters.
def apply_gradient(model, gradient, learning_rate):
    parameters = list(model.parameters())
    total_parameters = sum(param.numel() for param in parameters)

    if gradient.ndim != 1:
        raise ValueError("Expected a 1D aggregated gradient.")

    if gradient.numel() != total_parameters:
        raise ValueError(f"Gradient has {gradient.numel()} elements, "f"but model has {total_parameters} parameters.")
        
    offset = 0                                              # Keeps track of the position in the flattened gradient tensor

    with torch.no_grad():
        for param in model.parameters():
            num_elements = param.numel()                    # Get the total number of elements in the current parameter tensor

            # Extract the corresponding slice of the flattened gradient tensor for the current parameter
            param_gradient = gradient[offset:offset + num_elements].reshape(param.shape) # reshape the slice to match the shape of the current parameter tensor

            param.sub_(learning_rate * param_gradient)      # Update the parameter in place by subtracting the scaled gradient (learning_rate * param_gradient) from the current parameter value

            offset += num_elements

    return model

# Approximate the geometric median using Weiszfeld iterations.
def geometric_median(gradients, max_iter=100, tol=1e-8, eps=1e-8):
    if gradients.ndim != 2:
        raise ValueError("Expected a 2D tensor of client gradients.")

    if gradients.shape[0] == 0:
        raise ValueError("At least one client gradient is required.")

    if not torch.isfinite(gradients).all():
        raise ValueError("Client gradients must be finite.")

    center = gradients.mean(dim=0)                      # Initialize with the arithmetic mean

    for i in range(max_iter):
        # Euclidean distance from each client gradient to the center
        distances = torch.linalg.vector_norm(gradients - center, dim=1)     # Subtracts the center from each client gradient, via broadcasting
                                                                            # Vector_norm computes the Euclidean norm (L2 norm) along the specified dimension (dim=1), 
                                                                            # resulting in a 1D tensor of distances from each client gradient to the current center

        zero_mask = distances <= eps                    # Identify gradients that coincide with the center

        r = zero_mask.sum().item()                      # Count the coinciding gradients

        nonzero_gradients = gradients[~zero_mask]       # Keep only gradients that do not coincide with the center
        nonzero_distances = distances[~zero_mask]

        if nonzero_gradients.shape[0] == 0:             # If all gradients coincide with the center, return it
            return center

        # Calculate the resultant direction
        directions = (nonzero_gradients - center) / nonzero_distances.unsqueeze(1)

        R = directions.sum(dim=0)

        gamma = torch.linalg.vector_norm(R).item()

        if gamma <= r:                                  # Check whether the current center is already optimal
            return center

        weights = 1.0 / nonzero_distances               # Calculate the standard Weiszfeld update

        T = (nonzero_gradients * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

        alpha = r / gamma                               # Calculate the correction factor

        new_center = alpha * center + (1 - alpha) * T   # Apply the modified Weiszfeld update

        # Check convergence
        if torch.linalg.vector_norm(new_center - center) < tol:
            return new_center

        center = new_center

    return center



def huber_aggregation(gradients, client_sizes, thresholds, max_iter=100, tol=1e-6, eps=1e-8):
    if gradients.ndim != 2:
        raise ValueError("Expected a 2D tensor of client gradients.")

    if gradients.shape[0] == 0:
        raise ValueError("At least one client gradient is required.")

    if not torch.isfinite(gradients).all():
        raise ValueError("Client gradients must be finite.")

    client_sizes = torch.as_tensor(client_sizes, device=gradients.device, dtype=gradients.dtype)

    thresholds = torch.as_tensor(thresholds, device=gradients.device, dtype=gradients.dtype)

    if client_sizes.ndim != 1 or client_sizes.numel() != gradients.shape[0]:
        raise ValueError("One client size is required per gradient.")

    if thresholds.ndim == 0:
        thresholds = thresholds.expand(gradients.shape[0])

    if thresholds.ndim != 1 or thresholds.numel() != gradients.shape[0]:
        raise ValueError("One threshold is required per gradient.")

    if not torch.isfinite(client_sizes).all() or (client_sizes <= 0).any():
        raise ValueError("Client sizes must be finite and positive.")

    if not torch.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError("Thresholds must be finite and positive.")

    center = (gradients * client_sizes.unsqueeze(1)).sum(dim=0) / client_sizes.sum()

    for iteration in range(max_iter):
        distances = torch.linalg.vector_norm(gradients - center, dim=1)

        safe_distances = distances.clamp_min(eps)

        huber_weights = torch.minimum(torch.ones_like(distances), thresholds / safe_distances)

        weights = client_sizes * huber_weights

        new_center = (gradients * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

        if torch.linalg.vector_norm(new_center - center) < tol:
            return new_center

        center = new_center

    return center


def huber_aggregation_paper(gradients, thresholds,max_iter=100, tol=1e-6, eps=1e-8):
    if gradients.ndim != 2:
        raise ValueError("Expected a 2D tensor of client gradients.")

    if gradients.shape[0] == 0 or gradients.shape[1] == 0:
        raise ValueError("Gradients must contain clients and parameters.")

    if not torch.isfinite(gradients).all():
        raise ValueError("Client gradients must be finite.")

    thresholds = torch.as_tensor(thresholds, device=gradients.device, dtype=gradients.dtype)

    if thresholds.ndim == 0:
        thresholds = thresholds.expand(gradients.shape[0])

    if thresholds.ndim != 1 or thresholds.numel() != gradients.shape[0]:
        raise ValueError("One threshold is required per gradient.")

    if not torch.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError("Thresholds must be finite and positive.")

    if max_iter <= 0 or tol <= 0 or eps <= 0:
        raise ValueError("max_iter, tol and eps must be positive.")

    # Initialize the center using the arithmetic mean.
    center = gradients.mean(dim=0)

    for iteration in range(max_iter):
        # Calculate the distance of each gradient from the current center.
        distances = torch.linalg.vector_norm(gradients - center, dim=1)

        # Calculate the Huber weights.
        safe_distances = distances.clamp_min(eps)

        weights = torch.minimum(torch.ones_like(distances), thresholds / safe_distances)

        # If a gradient coincides with the center, its weight is 1.
        weights = torch.where(distances <= eps, torch.ones_like(weights), weights)

        # Update the center using the weighted average.
        new_center = (gradients * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

        # Check convergence.
        if torch.linalg.vector_norm(new_center - center) < tol:
            return new_center

        center = new_center

    return center
