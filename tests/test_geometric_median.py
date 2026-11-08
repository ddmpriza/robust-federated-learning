import torch

from src.gradient_aggregation import geometric_median


gradients = torch.tensor([
    [2.0, 0.0],
    [4.0, 0.0],
    [100.0, 0.0]
])

mean = gradients.mean(dim=0)

median = geometric_median(gradients)

expected = torch.tensor([4.0, 0.0])

print("Arithmetic mean:", mean)
print("Geometric median:", median)

print(
    "Test passed:",
    torch.allclose(median, expected, atol=1e-3)
)