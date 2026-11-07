import torch
from src.gradient_aggregation import gradient_mean

gradients = torch.tensor([
    [2.0, 4.0],
    [6.0, 8.0],
    [10.0, 12.0]
])

client_sizes = torch.tensor([100, 100, 200])

result = gradient_mean(gradients, client_sizes)

print("Aggregated gradient:", result)
print("Test passed:", torch.allclose(
    result, torch.tensor([7.0, 9.0])
))