
import torch

from src.data import load_mnist
from src.model import MNISTModel
from src.client import compute_client_gradient

train_dataset, _ = load_mnist()

model = MNISTModel()

weights_before = [
    p.detach().clone() for p in model.parameters()
]

gradient = compute_client_gradient(
    model=model,
    dataset=train_dataset,
    batch_size=32
)

print("Gradient shape:", gradient.shape)
print("Gradient norm:", torch.linalg.vector_norm(gradient).item())
print("All values finite:", torch.isfinite(gradient).all().item())

weights_unchanged = all(
    torch.equal(before, after)
    for before, after in zip(weights_before, model.parameters())
)

print("Model unchanged:", weights_unchanged)
