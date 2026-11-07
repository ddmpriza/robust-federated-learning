import torch
from torch.utils.data import TensorDataset, Subset

from src.model import MNISTModel
from src.client import compute_client_gradient
from src.gradient_aggregation import gradient_mean


torch.manual_seed(42)

# Create a small synthetic dataset
images = torch.randn(12, 1, 28, 28)
labels = torch.randint(0, 10, (12,))

dataset = TensorDataset(images, labels)

# Divide data among three clients with unequal sizes
clients = [
    Subset(dataset, range(0, 2)),
    Subset(dataset, range(2, 7)),
    Subset(dataset, range(7, 12))
]

model = MNISTModel()

# Federated gradient computation
client_gradients = []
client_sizes = []

for client in clients:
    gradient = compute_client_gradient(model, client, batch_size=4)

    client_gradients.append(gradient)
    client_sizes.append(len(client))

federated_gradient = gradient_mean(
    gradients=torch.stack(client_gradients),
    client_sizes=torch.tensor(client_sizes)
)

# Centralized gradient computation
centralized_gradient = compute_client_gradient(
    model=model,
    dataset=dataset,
    batch_size=4
)

# Compare
print(
    "Federated and centralized gradients match:",
    torch.allclose(
        federated_gradient,
        centralized_gradient,
        atol=1e-6,
        rtol=1e-5
    )
)

print(
    "Maximum absolute difference:",
    (federated_gradient - centralized_gradient).abs().max().item()
)